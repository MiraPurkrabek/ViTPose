#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29503}

# Select a random port in range [29500, 29530]
PORT=$(shuf -i 29500-63530 -n 1)

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test_naive.py $CONFIG $CHECKPOINT --n-gpus $GPUS --launcher pytorch ${@:4}
