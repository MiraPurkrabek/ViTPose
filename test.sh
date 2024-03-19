CFG="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_small_coco_256x192.py"
# CFG="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py"
# CFG="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_small_coco_256x192_reproduce_OAMVal.py"

# CFG="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_small_coco_256x192_blackout_unfreeze.py"

# CFG="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_small_coco_256x192_blackout_unfreeze_wprob.py"

# CFG="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_small_coco_256x192_errEstimate.py"
# CFG="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_small_coco_256x192_errEstimate_fromHTM.py"

# CFG="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_small_coco_256x192_full_blackout_finetune.py"
# CFG="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_small_coco_256x192_full_fromHTM_blackout_finetune.py"

# CFG="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192_udp_fromHTM.py"

#################################################
#################################################

WEIGHTS="models/pretrained/vitpose-s.pth"
# WEIGHTS="models/pretrained/vitpose-b.pth"

# WEIGHTS="work_dirs/ViTPose_small_coco_256x192_blackout/best_AP_epoch_50.pth"
# WEIGHTS="work_dirs/ViTPose_small_coco_256x192_blackout_lre4/best_AP_epoch_33.pth"
# WEIGHTS="work_dirs/ViTPose_small_coco_256x192_blackout_lre5/best_AP_epoch_31.pth"
# WEIGHTS="work_dirs/ViTPose_small_coco_256x192_blackout_unfreeze/best_AP_epoch_4.pth"

# WEIGHTS="work_dirs/ViTPose_small_coco_256x192_blackout_unfreeze_wprob/epoch_50.pth"
# WEIGHTS="work_dirs/ViTPose_small_coco_256x192_blackout_unfreeze_wprob/epoch_50_fix.pth"
# WEIGHTS="work_dirs/ViTPose_small_coco_256x192_blackout_unfreeze_wprob/epoch_10_zeros.pth"

# WEIGHTS="work_dirs/ViTPose_small_coco_256x192_errEstimate/epoch_50_smoothL1.pth"
# WEIGHTS="work_dirs/ViTPose_small_coco_256x192_errEstimate/epoch_50_MSE.pth"
# WEIGHTS="work_dirs/ViTPose_small_coco_256x192_errEstimate_fromHTM/epoch_50_MSE.pth"

# WEIGHTS="work_dirs/ViTPose_small_coco_256x192_full_blackout_finetune/best_AP_epoch_91.pth"
# WEIGHTS="work_dirs/ViTPose_small_coco_256x192_full_blackout_finetune/zeros_best_AP_epoch_84.pth"
# WEIGHTS="work_dirs/ViTPose_small_coco_256x192_full_blackout_finetune/cropped_best_AP_epoch_5.pth"
# WEIGHTS="work_dirs/ViTPose_small_coco_256x192_full_blackout_finetune/zeros_fix_epoch_110.pth"
# WEIGHTS="work_dirs/ViTPose_small_coco_256x192_full_blackout_finetune/detach_prob_best_AP_epoch_52.pth"
# WEIGHTS="work_dirs/ViTPose_small_coco_256x192_full_blackout_finetune/attach_best_AP_epoch_30.pth"
# WEIGHTS="work_dirs/ViTPose_small_coco_256x192_full_blackout_zeros_finetune_ignoreNoise/best_AP_epoch_54.pth"
# WEIGHTS="work_dirs/ViTPose_small_coco_256x192_full_finetune/best_AP_epoch_91.pth"
# WEIGHTS="work_dirs/ViTPose_small_coco_256x192_full_fromHTM_blackout_finetune/best_AP_epoch_87.pth"

# WEIGHTS="models/pretrained/hrnet_w48_coco_256x192_udp-2554c524_20210223.pth"

tools/dist_test.sh \
    $CFG \
    $WEIGHTS \
    1 \
    
    