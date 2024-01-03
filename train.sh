CFG="configs/body/2d_kpt_sview_rgb_img/out_of_image_heatmap/coco/OOI_ViTPose_small_coco_256x192.py"

tools/dist_train.sh \
    $CFG \
    2 \

    