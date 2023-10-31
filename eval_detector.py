'''
This script takes GT path and DT path as command line arguments and outputs the evaluation results
from the COCOEval library. The GT and DT paths should be in COCO format. The script also outputs
the results in a json file in the DT path.
'''
import argparse
import os
import json
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO


def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation on COCO dataset")
    parser.add_argument(
        "--gt_path", help="path to ground truth json file", required=True, type=str
    )
    parser.add_argument(
        "--dt_path", help="path to detection json file", required=True, type=str
    )
    # Boolean flag to specify whether to ignore classes
    parser.add_argument(
        "--ignore-classes",
        help="ignore classes when evaluating",
        action="store_true",
    )
    args = parser.parse_args()
    return args
    print(dt.dataset["annotations"][0].keys())



def main(args):
    # load GT and DT json files
    gt = COCO(args.gt_path)
    dt = COCO(args.dt_path)

    # Check if GT has 'iscrowd'. If not, set it to 0
    for ann in gt.dataset["annotations"]:
        if "iscrowd" not in ann.keys():
            ann["iscrowd"] = 0

    # Check if both GT and DT has 'area'
    for ann in dt.dataset["annotations"]:
        if "area" not in ann.keys():
            ann["area"] = ann["bbox"][2] * ann["bbox"][3]

    # If DT does not have a score, set it to 1
    for ann in dt.dataset["annotations"]:
        if "score" not in ann.keys():
            ann["score"] = 1

    # Set all classes of DT to 'human' if ignore_classes is True
    if args.ignore_classes:
        for ann in dt.dataset["annotations"]:
            ann["category_id"] = 1

    # run evaluation
    coco_eval = COCOeval(gt, dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == "__main__":
    args = parse_args()
    main(args)