CFG="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_simple_COMBO_finetune_256x192.py"

tools/dist_train.sh \
    $CFG \
    4 \

    