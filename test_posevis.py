import os
import json
import posevis
import numpy as np
import cv2

data = json.load(open(
    "/datagrid/personal/purkrmir/data/OOI_eval/coco_cropped/annotations/person_keypoints_val2017.json",
    "r"
))

id2img = {}
for img in data["images"]:
    id2img[img["id"]] = img

ann = None
for a in data["annotations"]:
    # if 'amodal_box' in a.keys() and 583763 == a["image_id"]:
    if 'amodal_box' in a.keys() and np.random.rand() < 0.1:
        ann = a
        break

image_path = id2img[ann["image_id"]]["file_name"]
image_path = os.path.join(
    "/datagrid/personal/purkrmir/data/OOI_eval/coco_cropped/val2017", image_path
)

img = posevis.pose_visualization(image_path, ann, show_bbox=True)
cv2.imwrite("test.png", img)