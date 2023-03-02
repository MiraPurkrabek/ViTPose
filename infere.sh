#!/usr/bin/env bash

IN_DATA="../data/FACIS/nonsensitive_data/"

DET_MODEL="htc"


#################
# DETECTION
#################

if [ "$DET_MODEL" == "htc" ]; then
    # SOTA det
    DET_CFG="../open-mmlab/mmpose-mira/configs/mmdet/htc/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco.py"
    DET_PTH="https://download.openmmlab.com/mmdetection/v2.0/htc/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco_20200312-946fd751.pth"
elif [ "$DET_MODEL" == "mask2former" ]; then
    # SOTA det
    DET_CFG="../open-mmlab/mmpose-mira/configs/mmdet/mask2former/mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco-panoptic.py"
    DET_PTH="https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco-panoptic/mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco-panoptic_20220407_104949-d4919c44.pth"
fi

POSE_CFG="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_large_simple_coco_256x192.py"
POSE_PTH="models/pretrained/vitpose-l-simple.pth"

OUT_DATA="$IN_DATA/output/DET_${DET_MODEL}_POSE_ViTPose"

python top_down_img_demo_with_mmdet.py \
    $DET_CFG \
    $DET_PTH \
    $POSE_CFG \
    $POSE_PTH \
    --img-root $IN_DATA \
    --out-img-root $OUT_DATA \


