#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.

CONFIG=$1
GPUS=$2
WEIGHTS=$3
PORT=${PORT:-29502}

# Select a random port in range [29500, 29530]
PORT=$(shuf -i 29500-63530 -n 1)

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

if [[ -z $WEIGHTS ]]
then
    python -m torch.distributed.launch \
        --nproc_per_node=$GPUS \
        --master_port=$PORT \
        $(dirname "$0")/train.py \
        $CONFIG \
        --launcher pytorch ${@:4}
else
    python -m torch.distributed.launch \
        --nproc_per_node=$GPUS \
        --master_port=$PORT \
        $(dirname "$0")/train.py \
        $CONFIG \
        --resume-from $WEIGHTS \
        --launcher pytorch ${@:4}
fi


        # --cfg-options model.load_from=$WEIGHTS \

