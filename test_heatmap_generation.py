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
from mmpose.models import build_posenet


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
    
    cfg = Config.fromfile("configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_small_coco_256x192_blackout_unfreeze_wprob.py")
    dataset = build_dataset(cfg.data.train)
    shutil.rmtree("TargetTest", ignore_errors=True)
    os.makedirs("TargetTest", exist_ok=True)
    for i in tqdm(range(10), ascii=True):
        idx = np.random.randint(0, len(dataset))
        print(idx)
        # idx = 52601
        results = dataset[idx]
        target = results["target"]

        all_targets = np.array(target).reshape(17, -1)
        print("Image {:d}:".format(i))
        for j, tgt in enumerate(all_targets):
            print("\tJoint {:d}".format(j), end="")
            
            if results["target_weight"][j] <= 0:
                print(" (ignored):")
            else:
                print(":")
            
            print("\t\tshape:     {}".format(tgt.shape))
            print("\t\tmin:       {}".format(tgt.min()))
            print("\t\tmean:      {}".format(tgt.mean()))
            print("\t\tmax:       {}".format(tgt.max()))
            print("\t\tsum:       {}".format(tgt.sum()))
            print("\t\tsum [:-1]: {}".format(tgt[:-1].sum()))
            print("\t\tval:       {}".format(tgt[-1]))

        
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
        
        try:
            joints = results["joints_3d"]
            K, H, W = target.shape
            center = np.array([W // 2, H // 2])
            scale =  np.array([W / 200.0, H / 200.0])
            preds, maxvals = keypoints_from_heatmaps(
                target.reshape(1, K, H, W),
                center.reshape(1, 2),
                scale.reshape(1, 2),
                unbiased=False,
                post_process='default',
                kernel=11,
                valid_radius_factor=0.0546875,
                use_udp=True,
                target_type='GaussianHeatmap')
            preds = preds.squeeze()
            maxvals = maxvals.squeeze()

            for j, (pred, joint) in enumerate(zip(preds, joints)):
                print("Joint {:02d}:".format(j))
                print("\tPred:  {} ({})".format(pred, maxvals[j]))
                print("\tGT:    {}".format(joint))
                print("\tError: {}".format(np.linalg.norm(pred - joint[:2])))
        except KeyError:
            pass

        for j, tgt in enumerate(target):
            orig_tgt = tgt.copy()

            if tgt.ndim == 1:
                probs = tgt[-1:]
                tgt = tgt[:-1]
                tgt = tgt.reshape(64, 48)


            # Resize to 256x192 and colorize
            tgt = cv2.resize(tgt, (192, 256))
            tgt = tgt.astype(np.float32)
            tgt -= tgt.min()
            mask = tgt > 0
            if not np.any(mask[:]):
                continue

            tgt /= tgt.max()
            tgt *= 255
            tgt = tgt.astype(np.uint8)
            tgt = cv2.applyColorMap(tgt, cv2.COLORMAP_JET)
            
            # Only overlay parts where heatmap is non-zero
            img_copy = deepcopy(img)
            img_copy[mask] = tgt[mask]
            img_copy = img_copy.astype(np.uint8)

            try:
                joint = joints[j, :2].astype(int)
                # print(joint, img_copy.shape, img_copy.max())
                # breakpoint()
                img_copy = cv2.drawMarker(
                    img_copy.copy(),
                    joint,
                    (0, 0, 0),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=5,
                    thickness=1,
                )
            except NameError:
                pass

            img = cv2.addWeighted(img_copy, 0.3, img, 0.7, 0)
            
            img_copy = cv2.copyMakeBorder(img_copy, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(100, 100, 100))
            if len(probs) == 1:
                probs = [probs[0], probs[0], probs[0], probs[0]]
            img_copy[:20, :20, :] = 255 * probs[0]
            img_copy[:20, -20:, :] = 255 * probs[1]
            img_copy[-20:, :20, :] = 255 * probs[2]
            img_copy[-20:, -20:, :] = 255 * probs[3]

            cv2.imwrite("TargetTest/{:02d}_img_heatmap_{:02d}.png".format(i, j), img_copy)

            # Overlay the heatmap
        cv2.imwrite("TargetTest/{:02d}_img_w_heatmaps.png".format(i), img)
        

    # model = build_posenet(cfg.model)
    # print(model)
    
    
