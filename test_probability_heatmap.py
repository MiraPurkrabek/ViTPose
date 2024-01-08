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

# type='TopDownGenerateTarget',
#         sigma=2,
#         encoding='UDP',
#         target_type=target_type,
#         inf_strip_size=0.1),

HEATMAP_TYPE = 'ProbabilityHeatmap'
# HEATMAP_TYPE = 'GaussianHeatmap'

IMG_SIZE = np.array([192, 256])
N = 6

def test_generate_target(input_data, save_dir="TargetTest", show=True):
    target_generator = TopDownGenerateTarget(
        sigma=2.0,
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

    return heatmaps, target


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
    
    cfg = Config.fromfile("configs/body/2d_kpt_sview_rgb_img/out_of_image_heatmap/coco/20_epochs/OOI_ViTs_20e.py")
    dataset = build_dataset(cfg.data.train)
    input_data = dataset[np.random.randint(0, len(dataset))]
    

    # kpts_gt = deepcopy(input_data["joints_3d"][:, :2])
    # scale = input_data["scale"]
    # center = input_data["center"]
    # vis = input_data["joints_3d_visible"][:, 0].squeeze()
    # n = len(kpts_gt)

    heatmaps, target = test_generate_target(input_data, show=True)
    kpts_gt = target["joints_3d"][:, :2]
    scale = target["scale"]
    center = target["center"]
    vis = target["joints_3d_visible"][:, 0].squeeze()
    kpts_gt[:, 0] = kpts_gt[:, 0] / (IMG_SIZE[0]-1.0) * (200 * scale[0]) + center[0] - (100 * scale[0])
    kpts_gt[:, 1] = kpts_gt[:, 1] / (IMG_SIZE[1]-1.0) * (200 * scale[1]) + center[1] - (100 * scale[1])
    print("kpts shaoe", kpts_gt.shape)
    n = len(kpts_gt)

    print("Center: ", center, input_data["center"])
    print("Scale: ", scale, input_data["scale"])
    print(target.keys())

    kpts_test = test_keypoints_from_heatmaps(heatmaps, target)
    kpts_test = kpts_test.squeeze()

    pt_dists = []
    for i in range(n):
        pt_dist = np.sqrt((kpts_gt[i, 0] - kpts_test[i, 0]) ** 2 + (kpts_gt[i, 1] - kpts_test[i, 1]) ** 2)
        if not vis[i].astype(int).item() == 0:
            pt_dists.append(pt_dist)
        print("({:6.1f}, {:6.1f}) --> ({:6.1f}, {:6.1f})\t{:d}, {:7.2f}".format(
            kpts_gt[i, 0].astype(float),
            kpts_gt[i, 1].astype(float),
            kpts_test[i, 0].astype(float),
            kpts_test[i, 1].astype(float),
            vis[i].astype(int).item(),
            pt_dist,
        ))
    
    print("Mean: {:.2f}".format(np.mean(pt_dists)))


    max_x = np.max(kpts_gt[:, 0])
    max_y = np.max(kpts_gt[:, 1])
    max_x = np.max([max_x, np.max(kpts_test[:, 0])])
    max_y = np.max([max_y, np.max(kpts_test[:, 1])])
    
    vis = vis.astype(int)
    kpts_gt = np.hstack((kpts_gt, np.expand_dims(vis, axis=1)))
    gt = {
        "keypoints": kpts_gt,
        "bbox": [
            center[0] - scale[0]*100, 
            center[1] - scale[1]*100, 
            center[0] + scale[0]*100, 
            center[1] + scale[1]*100, 
        ],
    }
    kpts_test = np.hstack((kpts_test, np.expand_dims(vis, axis=1)))
    test_img = cv2.imread(input_data["image_file"])
    test_img = pose_visualization(test_img, gt, format="coco", line_type="dashed", show_bbox=True, width_multiplier=3.0)
    test_img = pose_visualization(test_img, kpts_test, format="coco", line_type="solid", width_multiplier=3.0)
    cv2.imwrite("TargetTest/00_pose.png", test_img)
    
    intest_img = target["img"].numpy().transpose(1, 2, 0)
    intest_img -= np.min(intest_img)
    intest_img /= np.max(intest_img)
    intest_img *= 255
    intest_img = intest_img.astype(np.uint8)
    cv2.imwrite("TargetTest/00_pose_input.png", intest_img)
    
    blank_img = np.ones((int(max_y+1), int(max_x+1), 3), dtype=np.uint8)*255
    blank_img = pose_visualization(blank_img, gt, format="coco", line_type="dashed", show_bbox=True, width_multiplier=3.0)
    blank_img = pose_visualization(blank_img, kpts_test, format="coco", line_type="solid", width_multiplier=3.0)
    cv2.imwrite("TargetTest/00_pose_blank.png", blank_img)

    

    # print(kpts_test)