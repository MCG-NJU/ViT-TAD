import math
import os
from collections import OrderedDict

import torch
import torch.nn as nn
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models import create_model
from timm.utils import ModelEma
from torch.nn import ModuleList
from transformers import (VideoMAEFeatureExtractor,
                          VideoMAEForVideoClassification)

from .modeling_finetune import VisionTransformer
import utils_
from utils_ import NativeScalerWithGradNormCount as NativeScaler
from utils_ import multiple_samples_collate
from vedacore.misc import registry
from functools import partial

@registry.register_module('backbone')
class VideoMAE(nn.Module):

    def __init__(self,
                 model_name="vit_small_patch16_224",
                 nb_classes=400,
                 num_frames=16,
                 num_segments=1,
                 tubelet_size=2,
                 fc_drop_rate=0.0,
                 drop=0.0,
                 drop_path=0.1,
                 attn_drop_rate=0.0,
                 use_checkpoint=False,
                 use_mean_pooling=True,
                 init_scale=0.001,
                 finetune=None,
                 img_size=224,
                 scale_factor=1,
                 use_divide=False,
                 change_backbone=False,
                 model_key="model|module",
                 model_prefix="",
                 input_size=224,
                 freeze_bn=True,
                 freeze_bn_affine=True,
                 use_partition=False,
                 change_pe=False,
                 change_pe_2D=False,
                 use_temporal_pe=False,
                 glob_attn=[
                     False, False, False, False, False, False, False, False,
                     False, False, False, False
                 ],
                 kernel_size=3,
                 padding=1,
                 stride=1,
                 n_segment=128
                 ):  
        super(VideoMAE, self).__init__()

        self._freeze_bn = freeze_bn
        self._freeze_bn_affine = freeze_bn_affine

        self.model_name = model_name
        self.nb_classes = nb_classes
        self.num_frames = num_frames
        self.num_segments = num_segments
        self.tubelet_size = tubelet_size
        self.fc_drop_rate = fc_drop_rate
        self.drop = drop
        self.drop_path = drop_path
        self.attn_drop_rate = attn_drop_rate
        self.use_checkpoint = use_checkpoint
        self.use_mean_pooling = use_mean_pooling
        self.init_scale = init_scale
        self.finetune = finetune
        self.model_key = model_key
        self.model_prefix = model_prefix
        self.input_size = input_size
        self.img_size = img_size
        self.use_partition = use_partition
        self.change_pe = change_pe
        self.change_pe_2D = change_pe_2D
        self.scale_factor = scale_factor
        self.glob_attn = glob_attn
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.use_divide = use_divide
        self.change_backbone = change_backbone
        self.use_temporal_pe = use_temporal_pe
        self.n_segment = n_segment
        self.model = vit_base_patch16_224(
            pretrained=False,
            num_classes=self.nb_classes,
            all_frames=self.num_frames * self.num_segments,
            tubelet_size=self.tubelet_size,
            fc_drop_rate=self.fc_drop_rate,
            drop_rate=self.drop,
            drop_path_rate=self.drop_path,
            attn_drop_rate=self.attn_drop_rate,
            use_checkpoint=self.use_checkpoint,
            use_mean_pooling=self.use_mean_pooling,
            init_scale=self.init_scale,
            img_size=self.img_size,
            use_partition=self.use_partition,
            change_pe=self.change_pe,
            change_pe_2D=self.change_pe_2D,
            scale_factor=self.scale_factor,
            glob_attn=self.glob_attn,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
            use_divide=self.use_divide,
            use_temporal_pe=self.use_temporal_pe,
            n_segment=self.n_segment,
        )

        patch_size = self.model.patch_embed.patch_size
        print("Patch size = %s" % str(patch_size))
        self.window_size = (self.num_frames // 2,
                            self.input_size // patch_size[0],
                            self.input_size // patch_size[1])
        self.patch_size = patch_size

        checkpoint = torch.load(self.finetune, map_location='cpu')
        print("Load ckpt from %s" % self.finetune)
        checkpoint_model = None
        for model_key in self.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = self.model.state_dict()

        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()
        for key in all_keys:
            if key.startswith('backbone.'):
                new_dict[key[9:]] = checkpoint_model[key]
            elif key.startswith('encoder.'):
                new_dict[key[8:]] = checkpoint_model[key]
            else:
                new_dict[key] = checkpoint_model[key]
        checkpoint_model = new_dict

        # interpolate position embedding
        if 'pos_embed' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]  # channel dim
            num_patches = self.model.patch_embed.num_patches  #
            num_extra_tokens = self.model.pos_embed.shape[
                -2] - num_patches  # 0/1

            # height (== width) for the checkpoint position embedding
            orig_size = int((
                (pos_embed_checkpoint.shape[-2] - num_extra_tokens) //
                (self.num_frames // self.model.patch_embed.tubelet_size))**0.5)
            # height (== width) for the new position embedding
            new_size = int((
                num_patches //
                (self.num_frames // self.model.patch_embed.tubelet_size))**0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" %
                      (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                # B, L, C -> BT, H, W, C -> BT, C, H, W
                pos_tokens = pos_tokens.reshape(
                    -1, self.num_frames // self.model.patch_embed.tubelet_size,
                    orig_size, orig_size, embedding_size)
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size,
                                                embedding_size).permute(
                                                    0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens,
                    size=(new_size, new_size),
                    mode='bicubic',
                    align_corners=False)
                # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(
                    -1, self.num_frames // self.model.patch_embed.tubelet_size,
                    new_size, new_size, embedding_size)
                pos_tokens = pos_tokens.flatten(1, 3)  # B, L, C
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embed'] = new_pos_embed

        utils_.load_state_dict(
            self.model, checkpoint_model, prefix=self.model_prefix)

    def forward(self, x):
        out = self.model(x) 
        return out

    def train(self, mode=True):
        super(VideoMAE, self).train(mode)
        if self._freeze_bn and mode:
            for _, m in self.model.named_modules(
            ): 
                if isinstance(m, nn.BatchNorm3d):
                    m.eval()
                    if self._freeze_bn_affine:
                        m.weight.requires_grad_(False)
                        m.bias.requires_grad_(False)
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self._freeze_bn_affine:
                        m.weight.requires_grad_(False)
                        m.bias.requires_grad_(False)
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()
                    if self._freeze_bn_affine:
                        m.weight.requires_grad_(False)
                        m.bias.requires_grad_(False)

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 400, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }

def vit_base_patch16_224(pretrained=False,**kwargs):  
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model
