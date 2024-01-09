import os
import numpy as np
import cv2
from mmpose.datasets.pipelines.top_down_transform import TopDownGenerateTarget, bbox_xywh2cs, TopDownGetBboxCenterScale
from mmpose.core.evaluation.top_down_eval import keypoints_from_heatmaps
from mmpose.datasets import build_dataset
from mmpose.datasets import build_dataloader
from mmcv import Config
import torch
from copy import deepcopy

from posevis import pose_visualization


HEATMAP_TYPE = 'GaussianHeatmap'

def test_generate_target(input_data, save_dir="TargetTest", show=True):
    target_generator = TopDownGenerateTarget(
        sigma=2.0,
        encoding='UDP',
        target_type=HEATMAP_TYPE,
    )
    print(input_data.keys())
    target = target_generator(input_data)
    heatmaps = target["target"]
    if show:
        os.makedirs(save_dir, exist_ok=True)
        for i in range(heatmaps.shape[0]):
            hmp = (heatmaps[i, :, :]).astype(np.float32)
            sm = np.sum(hmp)
            if np.max(hmp) > 0:
                hmp /= np.max(hmp)
                hmp *= 255
            hmp = hmp.astype(np.uint8)
            # print("{:d}: ({:7.2f}, {:7.2f}) [{:d}] --> [{:3d} - {:3d}] \t sum: {:3.1f}".format(
            #     i,
            #     kpts[i, 0].astype(float),
            #     kpts[i, 1].astype(float),
            #     vis[i].item(),
            #     np.min(hmp).astype(int),
            #     np.max(hmp).astype(int),
            #     sm,
            # ))
            save_path = os.path.join(save_dir, "heatmap_{:03d}.png".format(i))
            cv2.imwrite(save_path, hmp)

    return heatmaps, target

if __name__ == "__main__":
    
    cfg = Config.fromfile("configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_small_coco_256x192_force_zeros.py")
    dataset = build_dataset(cfg.data.train)
    results = dataset[np.random.randint(0, len(dataset))]
    target = results["target"]
    visibilities = results["joints_3d_visible"]
    target_weight = results["target_weight"]

    print(target.shape)
    for w, v, tg in zip(target_weight, visibilities, target):
        print("weight:", w, "\tvisibility:", v, "\ttarget all zero:", np.all(tg == 0))

    # heatmaps, target = test_generate_target(input_data, show=True)
