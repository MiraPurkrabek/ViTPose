import os
import numpy as np
import cv2
from mmpose.datasets.pipelines.top_down_transform import TopDownGenerateTarget, bbox_xywh2cs, TopDownGetBboxCenterScale
from mmpose.core.evaluation.top_down_eval import keypoints_from_heatmaps
from mmpose.datasets import build_dataset
from mmpose.datasets import build_dataloader
from mmcv import Config
import torch

# type='TopDownGenerateTarget',
#         sigma=2,
#         encoding='UDP',
#         target_type=target_type,
#         inf_strip_size=0.1),

HEATMAP_TYPE = 'ProbabilityHeatmap'
# HEATMAP_TYPE = 'GaussianHeatmap'

IMG_SIZE = np.array([192, 256])
HEATMAP_SIZE = np.array([40, 53])
N = 6

def test_generate_target(input_data, save_dir="TargetTest", show=True):
    target_generator = TopDownGenerateTarget(
        sigma=2,
        encoding='UDP',
        target_type=HEATMAP_TYPE,
        inf_strip_size=0.1,
    )
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

    return heatmaps


def test_keypoints_from_heatmaps(heatmaps, results):
    if np.ndim(heatmaps) == 3:
        heatmaps = np.expand_dims(heatmaps, axis=0)
    center = results['center']
    scale = results['scale']

    if np.ndim(center) == 1:
        # Copy bbox along batch dimension
        center = np.tile(center, (heatmaps.shape[0], 1))
    if np.ndim(scale) == 1:
        # Copy bbox along batch dimension
        scale = np.tile(scale, (heatmaps.shape[0], 1))

    print("Center: ", center)
    print("Scale: ", scale)

    kpts, _ = keypoints_from_heatmaps(
        heatmaps,
        center,
        scale,
        unbiased=False,
        post_process='default',
        kernel=11,
        valid_radius_factor=0.0546875,
        use_udp=True,
        target_type=HEATMAP_TYPE,
        inf_strip_size=0.1
    )

    return kpts


def test_probability_heatmap(n=10, save_dir="test"):
    pass


if __name__ == "__main__":
    n = N
    
    cfg = Config.fromfile("configs/body/2d_kpt_sview_rgb_img/out_of_image_heatmap/coco/ViTPose_small_coco_256x192.py")
    dataset = build_dataset(cfg.data.train)
    input_data = dataset[0]

    kpts_gt = input_data["joints_3d"][:, :2]
    vis = input_data["joints_3d_visible"][:, 0].squeeze()
    n = len(kpts_gt)

    heatmaps = test_generate_target(input_data, show=True)

    kpts_test = test_keypoints_from_heatmaps(heatmaps, input_data)
    kpts_test = kpts_test.squeeze()

    for i in range(n):
        pt_dist = np.sqrt((kpts_gt[i, 0] - kpts_test[i, 0]) ** 2 + (kpts_gt[i, 1] - kpts_test[i, 1]) ** 2)
        print("({:6.1f}, {:6.1f}) --> ({:6.1f}, {:6.1f})\t{:d}, {:7.2f}".format(
            kpts_gt[i, 0].astype(float),
            kpts_gt[i, 1].astype(float),
            kpts_test[i, 0].astype(float),
            kpts_test[i, 1].astype(float),
            vis[i].astype(int).item(),
            pt_dist,
        ))

    # print(kpts_test)