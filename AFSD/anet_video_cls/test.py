import json
import multiprocessing as mp
import os
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import tqdm

from AFSD.anet_video_cls.BDNet import BDNet
from AFSD.common import videotransforms
from AFSD.common.anet_dataset import ANET_Dataset, get_video_info, load_json
from AFSD.common.config import config
from AFSD.common.segment_utils import softnms_v2
from vedatad.misc import get_root_logger


lock = threading.Lock()

num_classes = 2
conf_thresh = config["testing"]["conf_thresh"]
top_k = config["testing"]["top_k"]
nms_thresh = config["testing"]["nms_thresh"]
nms_sigma = config["testing"]["nms_sigma"]
clip_length = config["dataset"]["testing"]["clip_length"]
stride = config["dataset"]["testing"]["clip_stride"]
crop_size = config["dataset"]["testing"]["crop_size"]
checkpoint_path = config["testing"]["checkpoint_path"]
json_name = config["testing"]["output_json"]
output_path = config["testing"]["output_path"]
ngpu = config["ngpu"]
softmax_func = True
if not os.path.exists(output_path):
    os.makedirs(output_path)


thread_num = ngpu

classifiers = ["builtin", "cuhk", "gt"]

processes = []

norm_cfg = config["other_config"].norm_cfg
img_mean = torch.tensor(norm_cfg["mean"]).reshape(3, 1, 1, 1)
img_std = torch.tensor(norm_cfg["std"]).reshape(3, 1, 1, 1)

# setup logger
workspace, ckpt_name = os.path.split(checkpoint_path)
epoch = ckpt_name.split("-")[1].split(".")[0]
log_file = os.path.join(workspace, f"Test-epoch_{epoch}.log")
logger = get_root_logger(log_file=log_file, log_level="INFO")

video_infos = get_video_info(
        config["dataset"]["testing"]["video_info_path"], subset="validation"
    )
mp4_data_path = config["dataset"]["testing"]["video_mp4_path"]

if softmax_func:
    score_func = nn.Softmax(dim=-1)
else:
    score_func = nn.Sigmoid()

centor_crop = videotransforms.CenterCrop(crop_size)

video_list = list(video_infos.keys())
video_num = len(video_list)
per_thread_video_num = video_num // thread_num
sub_video_list = []
sub_video_list.append(video_list[0 * per_thread_video_num : 1 * per_thread_video_num])
sub_video_list.append(video_list[1 * per_thread_video_num : 2 * per_thread_video_num])
sub_video_list.append(video_list[2 * per_thread_video_num : 3 * per_thread_video_num])
sub_video_list.append(video_list[3 * per_thread_video_num :])

cuhk_data = load_json("data/annots/anet/cuhk_val_simp_share.json")
cuhk_data_score = cuhk_data["results"]
cuhk_data_action = cuhk_data["class"]
with open("data/annots/anet/action_name.txt") as f:
    classes = f.read().splitlines()

global result_dict_builtin, result_dict_cuhk, result_dict_gt

def sub_processor( pid, video_list,result_dict_builtin,result_dict_cuhk,result_dict_gt): #lock,
    text = "processor %d" % pid
    torch.cuda.set_device(pid+4)  
    net = BDNet(config["other_config"], training=False)
    net.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    net.eval().cuda()

    num_videos = 0
    correct_videos = 0

    np.random.shuffle(video_list)
    for i, video_name in enumerate(video_list):

        sample_count = int(video_infos[video_name]["frame_num"])
        sample_fps = video_infos[video_name]["fps"]
        duration = video_infos[video_name]["duration"]
        gt_action_id = video_infos[video_name]["annotations"][0]["label_id"]
        offsetlist = [0]
        if video_name[:2]=="v_":
            video_name_temp=video_name[2:]
        else:
            video_name_temp=video_name
        data = ANET_Dataset.load_frames(mp4_data_path, video_name_temp)
        if len(data) > sample_count:
            data = data[:sample_count]

        frames = data
        frames = np.transpose(frames, [3, 0, 1, 2])
        data = centor_crop(frames)
        data = torch.from_numpy(data.copy())

        output = []
        for cl in range(num_classes):
            output.append([])
        res = torch.zeros(num_classes, top_k, 3)

        for offset in offsetlist:
            clip = data[:, offset : offset + clip_length]
            clip = clip.cuda().float()

            if clip.size(1) < clip_length:
                tmp = torch.zeros(
                    [clip.size(0), clip_length - clip.size(1), crop_size, crop_size],
                    dtype=clip.dtype,
                    device=clip.device,
                )
                clip = torch.cat([clip, tmp], dim=1)
            clip = (clip - img_mean.cuda()) / img_std.cuda()
            clip = clip.unsqueeze(0)  # [B,C,T,H,W]
            with torch.no_grad():
                output_dict = net(clip)

            action_logits = output_dict["action_logits"]
            action_scores = torch.softmax(action_logits, dim=-1)[0]
            action_id = torch.argmax(action_logits, dim=-1)[0]
            action_score = float(action_scores[action_id])

            num_videos += 1
            if action_id == gt_action_id:
                correct_videos += 1

            loc, conf, priors = (
                output_dict["loc"],
                output_dict["conf"],
                output_dict["priors"][0],
            )
            prop_loc, prop_conf = output_dict["prop_loc"], output_dict["prop_conf"]
            center = output_dict["center"]
            loc = loc[0] 
            conf = score_func(conf[0]) 
            prop_loc = prop_loc[0] 
            prop_conf = score_func(prop_conf[0]) 
            center = center[0].sigmoid()  

            pre_loc_w = loc[:, :1] + loc[:, 1:]
            loc = 0.5 * pre_loc_w * prop_loc + loc
            decoded_segments = torch.cat(
                [
                    priors[:, :1] * clip_length - loc[:, :1],
                    priors[:, :1] * clip_length + loc[:, 1:],
                ],
                dim=-1,
            )
            decoded_segments.clamp_(min=0, max=clip_length)

            conf = (conf + prop_conf) / 2.0
            conf = conf * center
            conf = conf.view(-1, num_classes).transpose(1, 0)
            conf_scores = conf.clone()

            for cl in range(1, num_classes): 
                # c_mask = conf_scores[cl] > conf_thresh
                c_mask = conf_scores[cl] > 1e-9
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_segments)
                segments = decoded_segments[l_mask].view(-1, 2)
                segments = (segments + offset) / sample_fps
                segments = torch.cat([segments, scores.unsqueeze(1)], -1)

                output[cl].append(segments)

        sum_count = 0
        for cl in range(1, num_classes):
            if len(output[cl]) == 0:
                continue
            tmp = torch.cat(output[cl], 0)
            tmp, count = softnms_v2(tmp, sigma=nms_sigma, top_k=top_k, score_threshold=1e-9)
            res[cl, :count] = tmp
            sum_count += count

        flt = res.contiguous().view(-1, 3)
        flt = flt.view(num_classes, -1, 3)
        proposal_list = {"cuhk": [], "builtin": [], "gt": []}
        for cl in range(1, num_classes):

            tmp = flt[cl].contiguous()
            tmp = tmp[(tmp[:, 2] > 0).unsqueeze(-1).expand_as(tmp)].view(-1, 3)
            if tmp.size(0) == 0:
                continue
            tmp = tmp.detach().cpu().numpy()
            for i in range(tmp.shape[0]):
                start_time = max(0, float(tmp[i, 0]))
                end_time = min(duration, float(tmp[i, 1]))
                if end_time <= start_time:
                    continue

                for classifier in classifiers:
                    if classifier == "cuhk":
                        scores = torch.from_numpy(np.array(cuhk_data_score[video_name[2:]]))
                        class_names = cuhk_data_action
                    elif classifier == "builtin":
                        scores = torch.softmax(action_logits, dim=-1)[0]
                        class_names = classes
                    elif classifier == "gt":
                        class_name = classes[gt_action_id - 1]
                        score = 1
                    else:
                        raise ValueError(f"classifier:{classifier} not supported")

                    if classifier == "gt":
                        tmp_proposal = {}
                        tmp_proposal["label"] = class_name
                        tmp_proposal["score"] = float(tmp[i, 2] * score)
                        tmp_proposal["segment"] = [start_time, end_time]
                        proposal_list[classifier].append(tmp_proposal.copy())
                    else:
                        sorted_scores, ids = torch.sort(scores, descending=True)
                        top_n = 2
                        for score, id in zip(sorted_scores[:top_n], ids[:top_n]):
                            if classifier == "builtin":
                                id = id - 1
                            class_name = class_names[id]
                            tmp_proposal = {}
                            tmp_proposal["label"] = class_name
                            tmp_proposal["score"] = float(tmp[i, 2] * score)
                            tmp_proposal["segment"] = [start_time, end_time]
                            proposal_list[classifier].append(tmp_proposal.copy())

        result_dict_builtin[video_name[2:]] = proposal_list["builtin"]
        result_dict_cuhk[video_name[2:]] = proposal_list["cuhk"]
        result_dict_gt[video_name[2:]] = proposal_list["gt"]

    logger.info(f"Video Level Top-1 Accuracy: {correct_videos/num_videos}")

if __name__ == '__main__':
    mp.set_start_method('spawn')

    result_dict_builtin = mp.Manager().dict()
    result_dict_cuhk = mp.Manager().dict()
    result_dict_gt = mp.Manager().dict()

    for i in range(thread_num):
        p = mp.Process(target=sub_processor, args=(i, sub_video_list[i],result_dict_builtin,result_dict_cuhk,result_dict_gt)) #lock, 
        p.start()
        processes.append(p)

    for p in processes:
        p.join()



    output_dict_builtin = {
        "version": "ActivityNet-v1.3",
        "results": dict(result_dict_builtin),
        "external_data": {},
    }
    json_name_builtin = json_name.replace(".json", "-builtin.json")
    with open(os.path.join(output_path, json_name_builtin), "w") as out:
        json.dump(output_dict_builtin, out)

    output_dict_cuhk = {
        "version": "ActivityNet-v1.3",
        "results": dict(result_dict_cuhk),
        "external_data": {},
    }
    json_name_cuhk = json_name.replace(".json", "-cuhk.json")
    with open(os.path.join(output_path, json_name_cuhk), "w") as out:
        json.dump(output_dict_cuhk, out)

    output_dict_gt = {
        "version": "ActivityNet-v1.3",
        "results": dict(result_dict_gt),
        "external_data": {},
    }
    json_name_gt = json_name.replace(".json", "-gt.json")
    with open(os.path.join(output_path, json_name_gt), "w") as out:
        json.dump(output_dict_gt, out)
