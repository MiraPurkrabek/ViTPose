CFG="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_small_coco_256x192_full_blackout_zeros_finetune_ignoreNoise.py"

tools/dist_train.sh \
    $CFG \
    1 \

    