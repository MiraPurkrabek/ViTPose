CFG="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_large_simple_coco_256x192.py"
WEIGHTS="models/pretrained/vitpose-l-simple.pth"

tools/dist_test.sh \
    $CFG \
    $WEIGHTS \
    1 \
    
    