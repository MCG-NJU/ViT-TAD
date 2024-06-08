keep_ratio = 0.375
num_frames = 768
strides = [8, 16, 32, 64, 128, 256]
norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
num_actions = 201  # num actions + bg
backbone_lr_multi = 0.04

loss = dict(action_loss_weight=1.0)
expid = "anet"
chunk_size = 32
shift_inp = False 
more_augmentation = True

#Not used
membank_cfg = dict(
    enable=True,
    shift_inp=shift_inp,
    keep_ratio=keep_ratio,
    chunk_size=chunk_size,
    t_downsample=2,
    drop_mode="uniform_jitter",
    mem_bank_meta_file=f"data/tmp/anet/memory_mechanism/{expid}/feat_swinb_k600_256x256_768frames/meta.json",
    mem_bank_dir=f"data/tmp/anet/memory_mechanism/{expid}/feat_swinb_k600_256x256_768frames",
)

model = dict(
    backbone=dict(
        typename='VideoMAE',
        model_name="vit_base_patch16_224",
        num_frames=num_frames,
        use_checkpoint=True,
        img_size=160, 
        use_partition=True, 
        change_pe_2D=True,  
        use_divide=True,
        use_temporal_pe=True,
        n_segment=384, 
        finetune='./pretrained/vit-b.pth',
        glob_attn = [False, False, 'attn_global_residual', False, False, 'attn_global_residual', False, False, 'attn_global_residual', False, False, 'attn_global_residual'],
    ),
    neck=[
        dict(
            typename="SRMSwin",
            srm_cfg=dict(
                in_channels=768,
                out_channels=768, 
                with_transformer=False,
            ),
        ),
        dict(
            typename="Transformer1DRelPos",
            encoder_layer_cfg=dict(
                dim=768, 
                num_heads=16,
                max_seq_len=num_frames // strides[0],
                drop_path=0.1,
            ),
            num_layers=3,
        ),
        dict(
            typename="TDM",
            in_channels=768,
            stage_layers=[1] * (len(strides) - 1),
            out_channels=768,
            conv_cfg=dict(typename="Conv1d"),
            norm_cfg=dict(typename="GN", num_groups=32),
            act_cfg=dict(typename="ReLU"),
            out_indices=list(range(len(strides))),
        ),
        dict(
            typename="FPN",
            in_channels=[768] * len(strides),  
            out_channels=768,
            num_outs=len(strides),
            start_level=0,
            conv_cfg=dict(typename="Conv1d"),
            norm_cfg=dict(typename="GN", num_groups=32),
        ),
    ],
    action_head=dict(num_classes=num_actions, in_channels=768, num_layers=4),  
)
scheduler = dict(type="MultiStepLR", milestones=[8, 10], gamma=0.1)
