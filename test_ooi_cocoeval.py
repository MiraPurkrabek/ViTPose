from xtcocotools.coco import COCO
from mmpose.datasets.datasets.top_down._cocoeval import COCOeval
# from mmpose.datasets.datasets.top_down._cocoeval_orig import COCOeval

gt = COCO(
    # '/datagrid/personal/purkrmir/data/OCHuman/COCO-like/annotations/person_keypoints_val2017.json',
    '/datagrid/personal/purkrmir/data/COCO/original/annotations/person_keypoints_val2017.json',
    # '/datagrid/personal/purkrmir/data/COCO/original/tmp_annotations/person_keypoints_val2017_one.json',
)

pred = gt.loadRes(
    # '/datagrid/personal/purkrmir/ViTPose/coco_test/vanilla_OCHuman.json',
    '/datagrid/personal/purkrmir/ViTPose/coco_test/vanilla_COCO.json',
    # '/datagrid/personal/purkrmir/data/COCO/original/tmp_annotations/person_keypoints_val2017_pred.json',
)

coco_eval = COCOeval(gt, pred, 'keypoints', extended_oks=True, alpha=None)
coco_eval.evaluate()
coco_eval.accumulate()
ret = coco_eval.summarize()

print(ret)
    