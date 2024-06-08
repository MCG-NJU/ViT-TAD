import json
import math
import os
import random

import decord
import numpy as np
import torch
from torch.utils.data import Dataset
from vedatad.partial_feedback import memory_bank

from AFSD.anet_video_cls.anet_dataset import (annos_transform, get_video_info,
                                              split_videos)
from AFSD.anet_video_cls.membank.graddrop import generate_indices
from AFSD.common import videotransforms
from AFSD.common.config import config

import os.path as osp
import glob
import vedacore.fileio as fileio
import vedacore.image as image

cfg = config["other_config"]


class ANET_Dataset(Dataset):
    def __init__(
        self,
        video_info_path,
        video_dir,
        clip_length,
        crop_size,
        stride,
        channels=3,
        rgb_norm=True,
        training=True,
        norm_cfg=dict(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
        binary_class=False,
        more_augmentation=False,
    ):
        self.training = training
        subset = "training" if training else "validation"
        video_info = get_video_info(video_info_path, subset)
        self.training_list, self.th = split_videos(video_info, clip_length, stride,video_dir, binary_class)
        self.clip_length = clip_length
        self.crop_size = crop_size
        self.rgb_norm = rgb_norm
        self.video_dir = video_dir
        self.channels = channels
        self.norm_cfg = norm_cfg

        self.random_crop = videotransforms.RandomCrop(crop_size)
        self.random_flip = videotransforms.RandomHorizontalFlip(p=0.5)
        self.center_crop = videotransforms.CenterCrop(crop_size)

        self.more_augmentation = more_augmentation
        if self.more_augmentation:
            self.photo_distortion = videotransforms.PhotoMetricDistortion()
            self.random_rotate = videotransforms.Rotate(
                limit=(-45, 45), border_mode="reflect101", p=0.5
            ) 
        if self.rgb_norm:
            self.img_mean = torch.tensor(self.norm_cfg["mean"]).reshape(3, 1, 1, 1)
            self.img_std = torch.tensor(self.norm_cfg["std"]).reshape(3, 1, 1, 1)

        if hasattr(cfg, "membank_cfg") and cfg.membank_cfg["enable"]:
            self.graddrop_enabled = True
            self.membank_cfg = cfg.membank_cfg
        else:
            self.graddrop_enabled = False

    def __len__(self):
        return len(self.training_list)

    @staticmethod
    def load_video(video_dir, video_name: str):
        """load video as frames.

        Args:
            video_dir (str): The video directory.
            video_name (str): The video name.

        Returns: numpy.array. shape: (T,H,W,C).

        """
        path = os.path.join(video_dir, video_name + ".mp4")
        if os.path.isfile(path):
            vr = decord.VideoReader(path)
            data = vr.get_batch(range(len(vr))).asnumpy()
        else:
            path = os.path.join(video_dir, video_name + ".npy")
            data = np.load(path)
        return data
    
    @staticmethod
    def load_frames(video_dir, video_name: str):
        """load video as frames.

        Args:
            video_dir (str): The video directory.
            video_name (str): The video name.

        Returns: numpy.array. shape: (T,H,W,C).

        """
        import mmcv

        video_name = osp.join(video_dir,video_name)
        imgfiles = sorted(glob.glob(osp.join(video_name, '*')))
        num=len(imgfiles)
        file_client = fileio.FileClient(**dict(backend='disk'))
        imgs = []
        for img_id in range(num):
            img_id=int(img_id)
            filename= imgfiles[img_id]
            img_bytes = file_client.get(filename)
            img = image.imfrombytes(img_bytes, flag='color')
            img = img.astype(np.float32)
            img_h, img_w = img.shape[:2]
            frame_resize=(180,-1) 
            max_long_edge = max(frame_resize)
            max_short_edge = min(frame_resize)
            if max_short_edge == -1:
                # assign np.inf to long edge for rescaling short edge later.
                frame_resize = (np.inf, max_long_edge)
            new_w, new_h = mmcv.rescale_size((img_w, img_h), frame_resize)
            img = mmcv.imresize(img, (new_w, new_h), interpolation="bilinear")

            imgs.append(img)
        imgs = np.array(imgs)
        return imgs


    def collate_fn(self, batch):
        clips = []
        targets = []
        scores = []
        metas = []

        for sample in batch:
            clips.append(sample[0])
            targets.append(torch.FloatTensor(sample[1]))
            scores.append(sample[2])
            metas.append(sample[3])

        clips = torch.stack(clips, 0)  # B,C,T,H,W
        scores = torch.stack(scores, 0)

        if not self.graddrop_enabled:
            return (clips, targets, scores, None)

        # for partial feedback
        chunk_size = cfg.chunk_size
        t_downsample = self.membank_cfg["t_downsample"]
        num_chunks = cfg.num_frames // cfg.chunk_size
        drop_features = []

        if cfg.shift_inp:
            # pad frames at begining and end. The number of padded frames is chunk_size.
            frame_pad_l = int(torch.randint(0, chunk_size, size=[]))
            keep_ratio = (self.membank_cfg["keep_ratio"] * num_chunks) / (num_chunks + 1)
            num_chunks += 1
        else:
            frame_pad_l = 0
            keep_ratio = self.membank_cfg["keep_ratio"]

        keep_idx, drop_idx = generate_indices(
            num_chunks, keep_ratio, mode=self.membank_cfg["drop_mode"]
        )

        for meta in metas:
            video_name = meta["video_name"]
        gd = {
            "metas": metas,
        }
        return (clips, targets, scores, gd) 

    def __getitem__(self, idx):
        sample_info = self.training_list[idx]
        video_name = sample_info["video_name"]
        offset = sample_info["offset"]
        annos = sample_info["annos"]
        frame_num = sample_info["frame_num"]
        th = int(self.th[sample_info["video_name"]] / 4)
        metas = {"video_name": video_name[2:], "num_frames": frame_num} 
        if video_name[:2]=="v_":
            video_name=video_name[2:]
        data = self.load_frames(self.video_dir, video_name) 
        start = offset
        end = min(offset + self.clip_length, frame_num)
        frames = data[start:end]
        frames = np.transpose(frames, [3, 0, 1, 2])

        c, t, h, w = frames.shape
        if t < self.clip_length:
            pad_t = self.clip_length - t
            zero_clip = np.zeros([c, pad_t, h, w], dtype=frames.dtype)
            frames = np.concatenate([frames, zero_clip], 1)

        # random crop and flip
        if self.training:
            frames = self.random_flip(self.random_crop(frames))
            if self.more_augmentation:
                frames = frames.astype(np.float32)
                frames = self.photo_distortion(frames)
                frames = self.random_rotate(frames)
        else:
            frames = self.center_crop(frames)

        input_data = torch.from_numpy(frames.copy()).float()
        if self.rgb_norm:
            input_data = (input_data - self.img_mean) / self.img_std
        annos = annos_transform(annos, self.clip_length)
        target = np.stack(annos, 0)

        scores = np.stack(
            [sample_info["action"], sample_info["start"], sample_info["end"]], axis=0
        )
        scores = torch.from_numpy(scores.copy()).float()

        return input_data, target, scores, metas
