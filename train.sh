CFG="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_small_combo_true_finetune_3kBOTTOM_rotated_256x192.py"

tools/dist_train.sh \
    $CFG \
    4 \

    