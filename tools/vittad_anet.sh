export MASTER_ADDR=localhost
export MASTER_PORT=12447 # any unused port.

expid=anet
# make sure the expid in the additional_config file matches the expid here.
additional_config=AFSD/anet_video_cls/configs/membank/$expid.py

config=AFSD/anet_video_cls/configs/anet.yaml
ckpt_path=workdir/vittad/${expid}


# do training
python3 AFSD/anet_video_cls/membank/train_membank.py $config --lw=1 --cw=1 --piou=0.6 --ssl 0. \
    --checkpoint_path $ckpt_path --addi_config $additional_config \
    --learning_rate 1e-4 \
    --batch_size 1 \
    --ngpu 4 \

# do testing
for epoch in 10; do   
    echo "Testing Epoch: $epoch"
    output_json=${expid}-epoch_${epoch}-anet_rgb.json
    echo $output_json
    python3 AFSD/anet_video_cls/test.py $config  --nms_sigma=0.85 --ngpu=4 \
        --checkpoint_path $ckpt_path/checkpoint-$epoch.ckpt  \
        --output_json=$output_json \
        --addi_config $additional_config
    for classifier in "builtin" "cuhk" "gt"; do
        output_json=${expid}-epoch_${epoch}-anet_rgb-${classifier}.json
        python3 AFSD/anet_video_cls/eval.py output/$output_json \
            --workspace $ckpt_path --epoch ${epoch}_${classifier}
    done
done