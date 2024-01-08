# CFG="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_test_384x288.py"
# WEIGHTS="models/pretrained/hrnet_w48_coco_384x288-314c8528_20200708.pth"

CFG="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_small_coco_256x192.py"
# CFG="configs/body/2d_kpt_sview_rgb_img/out_of_image_heatmap/coco/OOI_ViTPose_small_coco_256x192_wo_flip.py"
# CFG="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py"
# CFG="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py"
# CFG="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_large_simple_synthetic_256x192.py"

WEIGHTS="models/pretrained/vitpose-s.pth"
# WEIGHTS="models/pretrained/vitpose-h.pth"
# WEIGHTS="models/pretrained/vitpose-b-multi-coco.pth"
# WEIGHTS="models/pretrained/vitpose-h-multi-coco.pth"

# WEIGHTS="work_dirs/ViTPose_huge_combo_finetune_3kBOTOM_rotated_256x192/best_AP_epoch_17.pth"
# WEIGHTS="work_dirs/ViTPose_huge_combo_finetune_3kBOTOM_rotated_256x192_lr5e5/best_AP_epoch_17.pth"

# WEIGHTS="work_dirs/ViTPose_small_combo_finetune_NoData_RCI_256x192/best_AP_epoch_853.pth"
# WEIGHTS="work_dirs/ViTPose_small_combo_finetune_3kTOP_RCI_2_256x192/best_AP_epoch_495.pth"
# WEIGHTS="work_dirs/ViTPose_small_combo_finetune_3kTOP_rotated_256x192/best_AP_epoch_463.pth"
# WEIGHTS="work_dirs/ViTPose_small_combo_finetune_10k_TOP_256x192/best_AP_epoch_358.pth"
# WEIGHTS="work_dirs/ViTPose_small_combo_finetune_10kTOP_fix_256x192/best_AP_epoch_489.pth"
# WEIGHTS="work_dirs/ViTPose_small_coco_rotated_256x192/best_AP_epoch_130.pth"
# WEIGHTS="work_dirs/ViTPose_small_combo_finetune_PoseFES2_256x192/best_AP_epoch_325.pth"
# WEIGHTS="work_dirs/ViTPose_small_combo_finetune_3kTOP_rotated_TOPval_256x192/best_AP_epoch_223.pth"

# WEIGHTS="work_dirs/ViTPose_small_coco_BOTTOM_256x192/best_AP_epoch_335.pth"
# WEIGHTS="work_dirs/ViTPose_small_coco_rotated_BOTTOM_256x192/best_AP_epoch_240.pth"
# WEIGHTS="work_dirs/ViTPose_small_combo_finetune_3kBOTTOM_256x192/best_AP_epoch_150.pth"
# WEIGHTS="work_dirs/ViTPose_small_combo_finetune_3kBOTTOM_rotated_256x192/best_AP_epoch_241.pth"
# WEIGHTS="work_dirs/ViTPose_small_combo_finetune_500BOTTOM_rotated_256x192/best_AP_epoch_388.pth"
# WEIGHTS="work_dirs/ViTPose_small_combo_finetune_5kBOTTOM_rotated_256x192/best_AP_epoch_144.pth"
# WEIGHTS="work_dirs/ViTPose_small_combo_finetune_1kBOTTOM_rotated_256x192/best_AP_epoch_312.pth"
# WEIGHTS="work_dirs/ViTPose_small_combo_finetune_1kBOTTOM_naked_rotated_256x192/best_AP_epoch_397.pth"
# WEIGHTS="work_dirs/ViTPose_small_combo_finetune_1kBOTTOM_bckg_rotated_256x192/best_AP_epoch_273.pth"
# WEIGHTS="work_dirs/ViTPose_small_combo_finetune_1kBOTTOM_armsup_rotated_256x192/best_AP_epoch_337.pth"
# WEIGHTS="work_dirs/ViTPose_small_combo_finetune_1kBOTTOM_256x192/best_AP_epoch_192.pth"
# WEIGHTS="work_dirs/ViTPose_small_combo_finetune_1kBOTTOM_extreme_rotated_256x192/best_AP_epoch_381.pth"
# WEIGHTS="work_dirs/ViTPose_small_combo_finetune_1kBOTTOM_dist_I_rotated_256x192/best_AP_epoch_378.pth"
# WEIGHTS="work_dirs/ViTPose_small_combo_finetune_1kBOTTOM_dist_II_rotated_256x192/best_AP_epoch_289.pth"
# WEIGHTS="work_dirs/ViTPose_small_combo_finetune_1kBOTTOM_dist_III_rotated_256x192/best_AP_epoch_364.pth"
# WEIGHTS="work_dirs/ViTPose_small_combo_finetune_1kBOTTOM_dist_IV_rotated_256x192/best_AP_epoch_402.pth"
# WEIGHTS="work_dirs/ViTPose_small_combo_finetune_3kBOTTOM_extreme_rotated_256x192/best_AP_epoch_133.pth"

# WEIGHTS="work_dirs/ViTPose_small_combo_finetune_500BOTTOM_rotated_amass_256x192/best_AP_epoch_302.pth"
# WEIGHTS="work_dirs/ViTPose_small_combo_finetune_1kBOTTOM_rotated_amass_256x192/best_AP_epoch_230.pth"
# WEIGHTS="work_dirs/ViTPose_small_combo_finetune_3kBOTTOM_rotated_amass_256x192/best_AP_epoch_412.pth"
# WEIGHTS="work_dirs/ViTPose_small_combo_finetune_500BOTTOM_HN_rotated_256x192/best_AP_epoch_236.pth"
# WEIGHTS="work_dirs/ViTPose_small_combo_finetune_1kBOTTOM_HN_rotated_256x192/best_AP_epoch_266.pth"
# WEIGHTS="work_dirs/ViTPose_small_combo_finetune_3kBOTTOM_HN_rotated_256x192/best_AP_epoch_771.pth"
# WEIGHTS="work_dirs/ViTPose_small_combo_finetune_5kBOTTOM_HN_rotated_256x192/best_AP_epoch_391.pth"
# WEIGHTS="work_dirs/ViTPose_small_combo_finetune_1kBOTTOM_truncatedGauss_rotated_256x192/best_AP_epoch_387.pth"

# WEIGHTS="work_dirs/ViTPose_small_coco_scratch_256x192/best_AP_epoch_300.pth"
# WEIGHTS="work_dirs/ViTPose_small_coco_scratch_rotated_256x192/best_AP_epoch_300.pth"

# WEIGHTS="work_dirs/ViTPose_small_coco_scratch_imitate_256x192/best_AP_epoch_260.pth"
# WEIGHTS="work_dirs/ViTPose_small_coco_size100_256x192/best_AP_epoch_290.pth"
# WEIGHTS="work_dirs/ViTPose_small_coco_size115_256x192/best_AP_epoch_300.pth"
# WEIGHTS="work_dirs/ViTPose_small_coco_size135_256x192/best_AP_epoch_280.pth"

# WEIGHTS="work_dirs/ViTPose_small_combo_finetune_3kTOPBOTTOM_rotated_256x192/best_AP_epoch_315.pth"

WEIGHTS="work_dirs/OOI_ViTPose_small_coco_256x192_wo_flip/best_AP_epoch_210.pth"

tools/dist_test.sh \
    $CFG \
    $WEIGHTS \
    2 \

    # --gpu-collect \
    
    