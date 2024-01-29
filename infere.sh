#!/usr/bin/env bash

# IN_DATA="../data/FACIS/NSFW_benchmark/"
# IN_DATA="../data/pose_experiments/MMA/"
# IN_DATA="../data/Contortionists/"
# IN_DATA="../data/Infants/anonymization/grant_request/"
# IN_DATA="../data/BOTTOM_VIEW_dataset/PoleVaultDuplantis_cut_frames_10fps/"
# IN_DATA="../smplx/sampled_poses/distance_2.0_simplicity_3.0_view_TOP_rotation_000/"
# IN_DATA="../data/OCHuman/tiny/"
# IN_DATA="../data/Floorball_SKV_data/camera_N_short/"
# IN_DATA="../data/CrowdedPose/COCO-like/"
# IN_DATA="../data/pose_experiments/stretch/"
# IN_DATA="../data/FACIS/bottom_view/"

DET_MODEL="htc"

#################
# DETECTION
#################

if [ "$DET_MODEL" == "htc" ]; then
    # SOTA det
    DET_CFG="../open-mmlab/mmdet/configs/htc/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco.py"
    DET_PTH="https://download.openmmlab.com/mmdetection/v2.0/htc/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco_20200312-946fd751.pth"
elif [ "$DET_MODEL" == "mask2former" ]; then
    # SOTA det
    DET_CFG="../open-mmlab/mmdet/configs/mask2former/mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco-panoptic.py"
    DET_PTH="https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco-panoptic/mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco-panoptic_20220407_104949-d4919c44.pth"
fi

# POSE_CFG="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_small_coco_256x192.py"
POSE_CFG="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py"
# POSE_PTH="models/pretrained/vitpose-s.pth"
POSE_PTH="models/pretrained/vitpose-h-multi-coco.pth"
# POSE_PTH="work_dirs/ViTPose_small_combo_finetune_3kTOP_rotated_256x192/best_AP_epoch_463.pth"
# POSE_PTH="work_dirs/ViTPose_small_combo_finetune_3kBOTTOM_rotated_256x192/best_AP_epoch_241.pth"
# POSE_PTH="work_dirs/ViTPose_huge_combo_finetune_3kBOTOM_rotated_256x192/best_AP_epoch_17.pth"


# OUT_DATA="$IN_DATA/output/DET_${DET_MODEL}_POSE_ViTPose-s"
# OUT_DATA="$IN_DATA/output/DET_${DET_MODEL}_POSE_ViTPose-s-RePoGen_bottom"
OUT_DATA="$IN_DATA/output/DET_${DET_MODEL}_POSE_ViTPose-h-multi-coco"
# OUT_DATA="$IN_DATA/output/DET_${DET_MODEL}_POSE_ViTPose-h-RePoGen_bottom"
# OUT_DATA="$IN_DATA/output/DET_manual_POSE_ViTPose-h"
# OUT_DATA="$IN_DATA/output/DET_${DET_MODEL}_POSE_3kTOP_rotated"

python top_down_img_demo_with_mmdet.py \
    $DET_CFG \
    $DET_PTH \
    $POSE_CFG \
    $POSE_PTH \
    --img-root $IN_DATA \
    --out-img-root $OUT_DATA

# python top_down_img_demo.py \
#     $POSE_CFG \
#     $POSE_PTH \
#     --img-root $IN_DATA/val2017/ \
#     --out-img-root $OUT_DATA \
#     --json-file $IN_DATA/annotations/person_keypoints_val2017.json \
#     # --json-file $IN_DATA/coco_annotations.json \
 


