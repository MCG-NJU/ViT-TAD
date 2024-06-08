import os
import os.path as osp
import random
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from vedatad.misc import get_root_logger

import AFSD.common.distributed as du
from AFSD.anet_video_cls.membank.membank_dataset import ANET_Dataset
from AFSD.anet_video_cls.membank.model import BDNetMemBank
from AFSD.anet_video_cls.multisegment_loss import (ActionClassLoss,
                                                   MultiSegmentLoss)
from AFSD.common.config import config
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

batch_size = config["training"]["batch_size"]
learning_rate = config["training"]["learning_rate"]
weight_decay = config["training"]["weight_decay"]
max_epoch = config["training"]["max_epoch"]
num_classes = 2
checkpoint_path = config["training"]["checkpoint_path"]
focal_loss = config["training"]["focal_loss"]
random_seed = config["training"]["random_seed"]
ngpu = config["ngpu"]

train_state_path = os.path.join(checkpoint_path, "training")
if not os.path.exists(train_state_path):
    os.makedirs(train_state_path)

resume = config["training"]["resume"]

cfg = config["other_config"]

def print_training_info(logger):
    logger.info(f"batch size: {batch_size}")
    logger.info(f"learning rate: {learning_rate}")
    logger.info(f"weight decay: {weight_decay}")
    logger.info(f"max epoch: {max_epoch}")
    logger.info(f"checkpoint path: {checkpoint_path}")
    logger.info(f"loc weight: {config['training']['lw']}")
    logger.info(f"cls weight: {config['training']['cw']}")
    logger.info(f"ssl weight: {config['training']['ssl']}")
    logger.info(f"piou: {config['training']['piou']}")
    logger.info(f"resume: {resume}")
    logger.info(f"gpu num: {ngpu}")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


GLOBAL_SEED = 1


def worker_init_fn(worker_id):
    set_seed(GLOBAL_SEED + worker_id)


def get_rng_states():
    states = []
    states.append(random.getstate())
    states.append(np.random.get_state())
    states.append(torch.get_rng_state())
    if torch.cuda.is_available():
        states.append(torch.cuda.get_rng_state())
    return states


def set_rng_state(states):
    random.setstate(states[0])
    np.random.set_state(states[1])
    torch.set_rng_state(states[2])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(states[3])


def save_model(epoch, model, optimizer):
    torch.save(
        model.module.state_dict(),
        os.path.join(checkpoint_path, "checkpoint-{}.ckpt".format(epoch)),
    )
    torch.save(
        {"optimizer": optimizer.state_dict(), "state": get_rng_states()},
        os.path.join(train_state_path, "checkpoint_{}.ckpt".format(epoch)),
    )


def resume_training(resume, model, optimizer):
    start_epoch = 1
    if resume > 0:
        start_epoch += resume
        model_path = os.path.join(checkpoint_path, "checkpoint-{}.ckpt".format(resume))
        model.module.load_state_dict(torch.load(model_path))
        train_path = os.path.join(train_state_path, "checkpoint_{}.ckpt".format(resume))
        state_dict = torch.load(train_path)
        optimizer.load_state_dict(state_dict["optimizer"])
        set_rng_state(state_dict["state"])
    return start_epoch


def calc_bce_loss(start, end, scores):
    start = torch.tanh(start).mean(-1)
    end = torch.tanh(end).mean(-1)

    loss_start = F.binary_cross_entropy(
        start.view(-1), scores[:, 1].contiguous().view(-1).cuda(), reduction="mean"
    )
    loss_end = F.binary_cross_entropy(
        end.view(-1), scores[:, 2].contiguous().view(-1).cuda(), reduction="mean"
    )
    return loss_start, loss_end


def forward_one_epoch(
    net, clips, targets, gd, CPD_Loss, action_class_loss, scores=None, training=True, ssl=False
):
    clips = clips.cuda()
    targets = [t.cuda() for t in targets]

    if training:
        output_dict = net(clips, gd, ssl=False)
    else:
        with torch.no_grad():
            output_dict = net(clips)

    if ssl:
        anchor, positive, negative = output_dict
        loss_ = []
        weights = [1, 0.1, 0.1]
        for i in range(3):
            loss_.append(nn.TripletMarginLoss()(anchor[i], positive[i], negative[i]) * weights[i])
        trip_loss = torch.stack(loss_).sum(0)
        return trip_loss
    else:
        loss_l, loss_c, loss_prop_l, loss_prop_c, loss_ct = CPD_Loss(
            [
                output_dict["loc"],
                output_dict["conf"],
                output_dict["prop_loc"],
                output_dict["prop_conf"],
                output_dict["center"],
                output_dict["priors"][0],
            ],
            targets,
        )
        loss_action, top1, zero_ratio = action_class_loss(
            output_dict["action_logits"], output_dict["priors"][0], targets
        )
        loss_start, loss_end = calc_bce_loss(output_dict["start"], output_dict["end"], scores)
        scores_ = F.interpolate(scores, scale_factor=1.0 / 8)  
        loss_start_loc_prop, loss_end_loc_prop = calc_bce_loss(
            output_dict["start_loc_prop"], output_dict["end_loc_prop"], scores_
        )
        loss_start_conf_prop, loss_end_conf_prop = calc_bce_loss(
            output_dict["start_conf_prop"], output_dict["end_conf_prop"], scores_
        )
        loss_start = loss_start + 0.1 * (loss_start_loc_prop + loss_start_conf_prop)
        loss_end = loss_end + 0.1 * (loss_end_loc_prop + loss_end_conf_prop)
        return (
            loss_l,
            loss_c,
            loss_prop_l,
            loss_prop_c,
            loss_ct,
            loss_start,
            loss_end,
            loss_action,
            top1,
            zero_ratio,
        )


def run_one_epoch(
    epoch,
    net,
    optimizer,
    data_loader,
    CPD_Loss,
    action_class_loss,
    training=True,
    logger=None,
):
    if training:
        assert logger is not None
        net.train()
    else:
        net.eval()

    loss_loc_val = 0
    loss_conf_val = 0
    loss_prop_l_val = 0
    loss_prop_c_val = 0
    loss_ct_val = 0
    loss_start_val = 0
    loss_end_val = 0
    loss_trip_val = 0
    top1_val = 0
    zero_ratio_val = 0
    cost_val = 0
    loss_action_val = 0

    rank = du.get_rank()

    if rank == 0:
        pbar = tqdm.tqdm(data_loader, ncols=0)
    else:
        pbar = data_loader

    for n_iter, (clips, targets, scores, gd) in enumerate(pbar):
        (
            loss_l,
            loss_c,
            loss_prop_l,
            loss_prop_c,
            loss_ct,
            loss_start,
            loss_end,
            loss_action,
            top1,
            zero_ratio,
        ) = forward_one_epoch(
            net,
            clips,
            targets,
            gd,
            CPD_Loss,
            action_class_loss,
            scores,
            training=training,
        )

        loss_l = loss_l * config["training"]["lw"]
        loss_c = loss_c * config["training"]["cw"]
        loss_prop_l = loss_prop_l * config["training"]["lw"]
        loss_prop_c = loss_prop_c * config["training"]["cw"]
        loss_ct = loss_ct * config["training"]["cw"]
        loss_action = loss_action * cfg["loss"]["action_loss_weight"]
        cost = (
            loss_l
            + loss_c
            + loss_prop_l
            + loss_prop_c
            + loss_ct
            + loss_start
            + loss_end
            + loss_action
        )

        if training:
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

        [
            loss_l,
            loss_c,
            loss_prop_l,
            loss_prop_c,
            loss_start,
            loss_end,
            loss_action,
            top1,
            zero_ratio,
            cost,
        ] = du.all_reduce(
            [
                loss_l,
                loss_c,
                loss_prop_l,
                loss_prop_c,
                loss_start,
                loss_end,
                loss_action,
                top1,
                zero_ratio,
                cost,
            ]
        )

        loss_loc_val += (loss_l.cpu().detach().numpy() - loss_loc_val) / (n_iter + 1)
        loss_conf_val += (loss_c.cpu().detach().numpy() - loss_conf_val) / (n_iter + 1)
        loss_prop_l_val += (loss_prop_l.cpu().detach().numpy() - loss_prop_l_val) / (n_iter + 1)
        loss_prop_c_val += (loss_prop_c.cpu().detach().numpy() - loss_prop_c_val) / (n_iter + 1)
        loss_ct_val += (loss_ct.cpu().detach().numpy() - loss_ct_val) / (n_iter + 1)
        loss_start_val += (loss_start.cpu().detach().numpy() - loss_start_val) / (n_iter + 1)
        loss_end_val += (loss_end.cpu().detach().numpy() - loss_end_val) / (n_iter + 1)
        cost_val += (cost.cpu().detach().numpy() - cost_val) / (n_iter + 1)
        loss_action_val += (loss_action.cpu().detach().numpy() - loss_action_val) / (n_iter + 1)
        top1_val += (top1.cpu().detach().numpy()[0] - top1_val) / (n_iter + 1)
        zero_ratio_val += (zero_ratio.cpu().detach().numpy() - zero_ratio_val) / (n_iter + 1)

        if rank == 0:
            pbar.set_postfix(
                loss="{:.5f}".format(cost_val),
                loss_l="{:.5f}".format(loss_loc_val),
                loss_c="{:.5f}".format(loss_conf_val),
                loss_prop_l="{:.5f}".format(loss_prop_l_val),
                loss_prop_c="{:.5f}".format(loss_prop_c_val),
                loss_ct="{:.5f}".format(loss_ct_val),
                loss_start="{:.5f}".format(loss_start_val),
                loss_end="{:.5f}".format(loss_end_val),
                loss_action="{:.5f}".format(loss_action_val),
                top1="{:.2f}%".format(top1_val),
                zero_ratio="{:.2f}%".format(zero_ratio_val * 100),
            )
    if training:
        prefix = "Train"
        if rank == 0:
            save_model(epoch, net, optimizer)
    else:
        prefix = "Val"

    if rank == 0:
        plog = "Epoch-{} {} Loss: Total - {:.5f}, loc - {:.5f}, conf - {:.5f}, prop_loc - {:.5f}, prop_conf - {:.5f}, IoU - {:.5f}, start - {:.5f}, end - {:.5f}, action - {:.5f}, top1 - {:.2f}%, zero_ratio - {:.2f}%".format(
            epoch,
            prefix,
            cost_val,
            loss_loc_val,
            loss_conf_val,
            loss_prop_l_val,
            loss_prop_c_val,
            loss_ct_val,
            loss_start_val,
            loss_end_val,
            loss_action_val,
            top1_val,
            zero_ratio_val * 100,
        )
        plog = plog + ", Triplet - {:.5f}".format(loss_trip_val)
        logger.info(plog)


def main(local_rank):
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(checkpoint_path, f"log.txt")
    logger = get_root_logger(log_file=log_file, log_level="INFO")
    if local_rank == 0:
        logger.info(f"====begin training at {timestamp}====")
    du.init_process_group(local_rank, ngpu, 0, 1, init_method="env://", dist_backend="nccl")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if local_rank == 0:
        print_training_info(logger)
    set_seed(random_seed)
    """
    Setup model
    """
    net = BDNetMemBank(config["other_config"]).to(device)

    # init weights
    if hasattr(config["other_config"], "pretrained_weights"):
        s = torch.load(config["other_config"]["pretrained_weights"])
        info = net.load_state_dict(s, strict=False)
        if local_rank == 0:
            logger.info(info)

    net = DDP(net, device_ids=[local_rank], find_unused_parameters=False) 

    """
    Setup optimizer
    """
    backbone_lr_multi = config["other_config"].get("backbone_lr_multi", 0.1)
    if local_rank == 0:
        logger.info(f"backbone_lr_multiplier: {backbone_lr_multi}")

    optimizer = torch.optim.Adam(
        [
            {
                "params": net.module.backbone.parameters(),
                "lr": learning_rate * backbone_lr_multi,
                "weight_decay": weight_decay,
            },
            {
                "params": net.module.coarse_pyramid_detection.parameters(),
                "lr": learning_rate,
                "weight_decay": weight_decay,
            },
        ]
    )

    # add scheduler
    if hasattr(cfg, "scheduler"):
        scheduler_cfg = cfg.scheduler.copy()
        type = scheduler_cfg.pop("type")
        if type == "MultiStepLR":
            scheduler = lr_scheduler.MultiStepLR(optimizer, **scheduler_cfg)
        else:
            raise ValueError(f"Not supported scheduler")
    else:
        scheduler = None

    """
    Setup loss
    """
    piou = config["training"]["piou"]
    CPD_Loss = MultiSegmentLoss(num_classes, piou, 1.0, use_focal_loss=focal_loss)
    action_class_loss = ActionClassLoss(cfg.num_actions)

    """
    Setup dataloader
    """
    more_augmentation = cfg.get("more_augmentation", False)
    logger.info(f"Dataset: more_augmentation:{more_augmentation}")
    train_dataset = ANET_Dataset(
        config["dataset"]["training"]["video_info_path"],
        config["dataset"]["training"]["video_mp4_path"],
        config["dataset"]["training"]["clip_length"],
        config["dataset"]["training"]["crop_size"],
        config["dataset"]["training"]["clip_stride"],
        channels=config["model"]["in_channels"],
        binary_class=True,
        norm_cfg=config["other_config"].norm_cfg,
        more_augmentation=more_augmentation,
    )
    sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=6,
        shuffle=False,
        worker_init_fn=worker_init_fn,
        collate_fn=train_dataset.collate_fn,
        pin_memory=False,
        drop_last=True,
        sampler=sampler,
    )

    """
    Start training
    """
    start_epoch = resume_training(resume, net, optimizer)

    for i in range(start_epoch, max_epoch + 1):
        sampler.set_epoch(i)
        run_one_epoch(
            i,
            net,
            optimizer,
            train_data_loader,
            CPD_Loss,
            action_class_loss,
            logger=logger,
        )
        if scheduler is not None:
            scheduler.step()


if __name__ == "__main__":
    mp.spawn(main, nprocs=ngpu, join=True)
