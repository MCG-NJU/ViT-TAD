
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import torch.utils.checkpoint as checkpoint
import math
import fvcore.nn.weight_init as weight_init
from vedacore.misc import registry,build_from_module

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 400, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channel, out_channel, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channel, out_channel, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm3d(out_channel)
        self.downsample = downsample
        self.stride = stride

        self.init_weight()

    def init_weight(self):
        torch.nn.init.xavier_uniform(self.conv2.weight)
        self.conv2.weight.data.fill_(0.0)
        self.conv2.bias.data.fill_(0.0)
        self.bn2.weight.data.fill_(1)
        self.bn2.bias.data.fill_(0.0)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention_global_residual(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.init_weight()

    def init_weight(self):
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()


    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None, global_atten=None, all_frames=256,kernel_size=3, padding=1, stride=1, n_segment=128):
        super().__init__()
        self.global_atten = global_atten
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.all_frames = all_frames

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

        if global_atten == True:
            self.basicblock = BasicBlock(in_channel=384, out_channel=384)
        if global_atten == "attn_global_residual" :
            self.attn_global_residual = Attention_global_residual(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)


    def forward(self, x):
        if self.global_atten == None or self.global_atten == True:
            if self.gamma_1 is None:
                x = x + self.drop_path(self.attn(self.norm1(x)))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
            else:
                x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        elif self.global_atten == False:
            frames = int(self.all_frames / 2)  
            num_chunks = int(frames / 8)  
            chunk_else = int(x.shape[1] / num_chunks) 
            x = x.reshape(x.shape[0], num_chunks, chunk_else, x.shape[-1]) 
            x = x.permute(1, 0, 2, 3)  

            y = torch.zeros(
                [x.shape[0]] + list(x.shape[1:]),
                dtype=x.dtype,
                device=x.device,
            )

            for i in range(x.shape[0]):
                if self.gamma_1 is None:
                    temp = x[i] + self.drop_path(self.attn(self.norm1(x[i]))) 
                    y[i] = temp + self.drop_path(self.mlp(self.norm2(temp)))
                else:
                    temp = x[i] + self.drop_path(self.gamma_1 * self.attn(self.norm1(x[i])))
                    y[i] = temp + self.drop_path(self.gamma_2 * self.mlp(self.norm2(temp)))
            x = y.permute(1, 0, 2, 3) 
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[-1])
        elif self.global_atten == "attn_global_residual":
            C = x.shape[-1]
            frames = int(self.all_frames / 2)  
            num_chunks = int(frames / 8)  
            chunk_else = int(x.shape[1] / num_chunks)
            width = int(math.sqrt(chunk_else / 8)) 
            bs = x.shape[0]
            x = x.reshape(x.shape[0], num_chunks, chunk_else, x.shape[-1])
            x = x.permute(1, 0, 2, 3)  
            y = torch.zeros(
                [x.shape[0]] + list(x.shape[1:]),
                dtype=x.dtype,
                device=x.device,
            )
            for i in range(x.shape[0]):
                if self.gamma_1 is None:
                    temp = x[i] + self.drop_path(self.attn(self.norm1(x[i])))
                    y[i] = temp + self.drop_path(self.mlp(self.norm2(temp)))
                else:
                    temp = x[i] + self.drop_path(self.gamma_1 * self.attn(self.norm1(x[i])))
                    y[i] = temp + self.drop_path(self.gamma_2 * self.mlp(self.norm2(temp)))
            x = y.permute(1, 0, 2, 3)  
            residual = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[-1])
            x = x.reshape(x.shape[0], x.shape[1], 8, width, width, x.shape[-1]) 
            x = x.permute(0, 3, 4, 1, 2, 5)  
            x = x.reshape(x.shape[0] * x.shape[1] * x.shape[2], x.shape[3] * x.shape[4], x.shape[5]) 
            x = self.attn_global_residual(x)  
            x = x.reshape(bs, width, width, num_chunks, 8, C)
            x = x.permute(0, 3, 4, 1, 2, 5)
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3] * x.shape[4], x.shape[5])
            x = x + residual
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (
                    num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim,
                              kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]),
                              stride=(self.tubelet_size, patch_size[0], patch_size[1]))

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x) 
        return x


# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 fc_drop_rate=0.,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 init_scale=0.,
                 all_frames=16,
                 tubelet_size=2,
                 use_checkpoint=False,
                 use_mean_pooling=True,
                 use_partition=False,
                 use_temporal_pe=False,
                 use_divide=False,
                 change_pe=False,
                 change_pe_2D=False,
                 scale_factor=1.0,
                 n_segment=128,
                 glob_attn=[False, False, False, False, False, False, False, False, False, False, False, False],
                 kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  
        self.tubelet_size = tubelet_size
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames,
            tubelet_size=self.tubelet_size)
        num_patches = 1568 
        self.num_patches = num_patches
        self.use_checkpoint = use_checkpoint
        self.scale_factor = scale_factor
        self.use_divide = use_divide
        self.use_temporal_pe = use_temporal_pe

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        if self.use_divide:
            self.pos_embed_temporal = nn.Parameter(torch.zeros(1, int(all_frames / 2), embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.glob_attn = glob_attn
        self.use_partition = use_partition
        self.change_pe = change_pe
        self.change_pe_2D = change_pe_2D
        if self.use_partition == True:
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    init_values=init_values, global_atten=self.glob_attn[i], all_frames=all_frames,
                    kernel_size=kernel_size, padding=padding, stride=stride,n_segment=n_segment)
                for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    init_values=init_values, global_atten=None, all_frames=all_frames, kernel_size=kernel_size,
                    padding=padding, stride=stride,n_segment=n_segment)
                for i in range(depth)])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_dropout = nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)
        if self.use_divide:
            trunc_normal_(self.pos_embed_temporal, std=.02)

        self.apply(self._init_weights)

        self.srm = build_from_module(dict(typename='AdaptiveAvgPool3d', output_size=(None, 1, 1)), nn)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes

    def forward_features(self, x):
        t = int(x.shape[-3] / 2) 
        h = int(x.shape[-2] / 16)
        w = int(x.shape[-1] / 16) 
        x = self.patch_embed(x) 

        B, C, _, _, _ = x.size() 
        x = x.flatten(2).transpose(1, 2) 

        if self.change_pe:
            pos_embed = torch.nn.functional.interpolate(
                self.pos_embed.permute(0, 2, 1), scale_factor=self.scale_factor, mode="linear", align_corners=False,
            )

        if self.change_pe_2D:
            pos_embed = self.pos_embed.reshape(-1,14,14,C).permute(0,3,1,2) 
            pos_embed = torch.nn.functional.interpolate(
                pos_embed, size=(w, h), mode='bicubic', align_corners=False) 
            pos_embed = pos_embed.reshape(1,8,C,w,h).permute(0,1,3,4,2)  
            pos_embed = pos_embed.reshape(1,pos_embed.shape[1]*pos_embed.shape[2]*pos_embed.shape[3],C).permute(0,2,1) 
            
        if self.use_partition:
            temp = pos_embed.clone()
            num = int(t / 8)
            for i in range(num - 1):
                pos_embed = torch.cat((pos_embed, temp), dim=2) 

        if self.use_temporal_pe:
            pos_embed_temporal = self.pos_embed_temporal.permute(0,2,1) 
            x = x.reshape(B, t, h, w, C)  
            x = x + pos_embed_temporal.permute(0, 2, 1).expand(B, -1, -1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1,h,w).permute(0, 1, 3, 4, 2).type_as(x).to(x.device).clone()
            x = x.reshape(B, x.shape[1] * x.shape[2] * x.shape[3], C) 

        if self.change_pe:
            if pos_embed is not None:
                x = x + pos_embed.permute(0, 2, 1).expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        elif self.change_pe_2D:
            if pos_embed is not None:
                x = x + pos_embed.permute(0, 2, 1).expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        else:
            if self.pos_embed is not None:
                x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = self.pos_drop(x) 
        if self.use_checkpoint:
            for blk in self.blocks:
                x = checkpoint.checkpoint(blk, x)
        else:
            for blk in self.blocks:
                x = blk(x)
        x = x.permute(0, 2, 1) 
        x = x.reshape(B, C, t, w, h)
        x = self.srm(x)
        x = x.squeeze(-1).squeeze(-1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x


# @register_model
def vit_small_patch16_224(pretrained=False, pretrained_cfg=None, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


# @register_model
def vit_base_patch16_224(pretrained=False,pretrained_cfg=None, pretrained_cfg_overlay=None,**kwargs): 
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


# @register_model
def vit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


# @register_model
def vit_large_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


# @register_model
def vit_large_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


# @register_model
def vit_large_patch16_512(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=512, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


# @register_model
def vit_huge_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


