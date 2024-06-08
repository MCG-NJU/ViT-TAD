#!/usr/bin/env bash

if (($# < 2)); then
    echo "Uasage: bash tools/trainval.sh config_file gpus"
    exit 1
fi
CONFIG="$1"
GPUS="$2"

IFS=', ' read -r -a gpus <<< ${GPUS}
NGPUS=${#gpus[@]}
PORT="$((29500 + RANDOM % 100))"

export CUDA_VISIBLE_DEVICES=${GPUS}

PYTHONPATH="$(dirname $0)/..":${PYTHONPATH} \
    NCCL_DEBUG=INFO python -m torch.distributed.launch --nproc_per_node=${NGPUS} --master_port=${PORT} \
        $(dirname "$0")/trainval.py $CONFIG --launcher pytorch ${@:3}
