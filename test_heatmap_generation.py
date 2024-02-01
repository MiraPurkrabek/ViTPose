import os
import shutil
import numpy as np
import cv2
from mmpose.datasets.pipelines.top_down_transform import TopDownGenerateTarget
from mmpose.core.evaluation.top_down_eval import keypoints_from_heatmaps
from mmpose.datasets import build_dataset
from mmpose.datasets import build_dataloader
from mmcv import Config
import torch
from copy import deepcopy
from tqdm import tqdm

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
    
    cfg = Config.fromfile("configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_small_coco_256x192_blackout.py")
    dataset = build_dataset(cfg.data.train)
    shutil.rmtree("TargetTest", ignore_errors=True)
    os.makedirs("TargetTest", exist_ok=True)
    for i in tqdm(range(30), ascii=True):
        idx = np.random.randint(0, len(dataset))
        # idx = 52601
        results = dataset[idx]
        target = results["target"]
        
        # If path to original image is available, save it
        try:
            image_path = results["image_file"]
            original_img = cv2.imread(image_path)
            cv2.imwrite("TargetTest/{:02d}_original_img.png".format(i), original_img)
        except KeyError:
            pass
        
        img = np.array(results["img"]).transpose(1, 2, 0)
        img -= np.min(img)
        img /= np.max(img)
        img *= 255
        img = img.astype(np.uint8)
        cv2.imwrite("TargetTest/{:02d}_img.png".format(i), img)
        for j, tgt in enumerate(target):
            
            # Resize to 256x192 and colorize
            tgt = cv2.resize(tgt, (192, 256))
            tgt = tgt.astype(np.float32)
            tgt *= 255
            tgt = tgt.astype(np.uint8)
            mask = tgt > 0
            tgt = cv2.applyColorMap(tgt, cv2.COLORMAP_JET)
            
            # Only overlay parts where heatmap is non-zero
            img_copy = deepcopy(img)
            img_copy[mask] = tgt[mask]

            # Overlay the heatmap
            img = cv2.addWeighted(img_copy, 0.3, img, 0.7, 0)
        
        cv2.imwrite("TargetTest/{:02d}_img_w_heatmaps.png".format(i), img)
    
    
