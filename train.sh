CFG="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_small_coco_256x192_sigm_out_force_known_zeros.py"

tools/dist_train.sh \
    $CFG \
    1 \

    