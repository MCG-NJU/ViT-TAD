import json
import os

import einops
import torch
import torch.nn.functional as F
from vedatad.partial_feedback.memory_bank import write_features

from AFSD.anet_video_cls.BDNet import BDNet


class BDNetMemBank(BDNet):

    """Docstring for ."""

    def __init__(self, cfg, training=True, frame_num=768):  
        """TODO: to be defined."""
        super().__init__(cfg, training, frame_num)
        self.chunk_size = cfg.chunk_size

        self.membank_cfg = cfg.membank_cfg
        self.shift_inp = cfg.membank_cfg.shift_inp

    def inference_backbone(self, x: torch.Tensor):
        return self.backbone(x)

    def forward_backbone(self, x: torch.Tensor, gd: dict = None):
        """TODO: Docstring for forward.

        Args:
            x (torch.Tensor): input frames. Shape: [B,C,T,H,W]
            gd (dict): The data contains the information for memory bank.
                {"keep_idx": keep_idx, # List[Int].
                "drop_idx": drop_idx,  # List[Int].
                "frozen_features": frozen_features, #Tensor. shape: [n_chunk, B, C, f_chunk_size]
                "metas": metas,}

        Returns: torch.Tensor. Bakcbone features with shape [B,C,T_f]

        """
        if gd is None:  # test mode
            return self.inference_backbone(x)

        backbone_feat = self.backbone(x)

        return backbone_feat

    def forward(self, x: torch.Tensor, gd: dict, ssl: bool = False):
        f = self.forward_backbone(x, gd)
        return self.forward_detector(f)
