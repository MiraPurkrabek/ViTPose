CFG="configs/body/2d_kpt_sview_rgb_img/out_of_image_heatmap/coco/20_epochs/ViTs_20e_BigHtm.py"

tools/dist_train.sh \
    $CFG \
    1 \

    