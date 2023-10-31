CFG="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_combo_finetune_3kBOTOM_rotated_256x192.py"

tools/dist_train.sh \
    $CFG \
    4 \

    