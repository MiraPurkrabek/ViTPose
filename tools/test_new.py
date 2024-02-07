# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import shutil
import os.path as osp
import warnings
import numpy as np
import cv2
from tqdm import tqdm

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmpose.apis import multi_gpu_test, single_gpu_test, vis_pose_result
from mmpose.datasets import build_dataloader, build_dataset, DatasetInfo
from mmpose.models import build_posenet
from mmpose.utils import setup_multi_processes

import json
import matplotlib.pyplot as plt

from posevis import pose_visualization

try:
    from mmcv.runner import wrap_fp16_model
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import wrap_fp16_model

def bbox_xywh2xyxy(bbox_xywh):
    """Transform the bbox format from xywh to x1y1x2y2.

    Args:
        bbox_xywh (ndarray): Bounding boxes (with scores),
            shaped (n, 4) or (n, 5). (left, top, width, height, [score])
    Returns:
        np.ndarray: Bounding boxes (with scores), shaped (n, 4) or
          (n, 5). (left, top, right, bottom, [score])
    """
    bbox_xyxy = bbox_xywh.copy()
    bbox_xyxy[:, 2] = bbox_xyxy[:, 2] + bbox_xyxy[:, 0]
    bbox_xyxy[:, 3] = bbox_xyxy[:, 3] + bbox_xyxy[:, 1]

    return bbox_xyxy

def bbox_cs2xywh(center, scale, padding=1., pixel_std=200.):
    """Transform the bbox format from (center, scale) to (x,y,w,h). Note that
    this is not an exact inverse operation of ``bbox_xywh2cs`` because the
    normalization of aspect ratio in ``bbox_xywh2cs`` is irreversible.

    Args:
        center (ndarray): Single bbox center in (x, y)
        scale (ndarray): Single bbox scale in (scale_x, scale_y)
        padding (float): Bbox padding factor that will be multilied to scale.
            Default: 1.0
        pixel_std (float): The scale normalization factor. Default: 200.0

    Returns:
        ndarray: Single bbox in (x, y, w, h)
    """

    wh = scale / padding * pixel_std
    xy = center - 0.5 * wh
    return np.r_[xy, wh]


def parse_args():
    parser = argparse.ArgumentParser(description='mmpose test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--n-gpus', type=int, help='number of used GPUs')
    parser.add_argument(
        '--work-dir', help='the dir to save evaluation results')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--eval',
        default=None,
        nargs='+',
        help='evaluation metric, which depends on the dataset,'
        ' e.g., "mAP" for MSCOCO')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def merge_configs(cfg1, cfg2):
    # Merge cfg2 into cfg1
    # Overwrite cfg1 if repeated, ignore if value is None.
    cfg1 = {} if cfg1 is None else cfg1.copy()
    cfg2 = {} if cfg2 is None else cfg2
    for k, v in cfg2.items():
        if v:
            cfg1[k] = v
    return cfg1


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))

    # dataset.oks_thr = 0.5

    # step 1: give default values and override (if exist) from cfg.data
    loader_cfg = {
        **dict(seed=cfg.get('seed'), drop_last=False, dist=distributed),
        **({} if torch.__version__ != 'parrots' else dict(
               prefetch_num=2,
               pin_memory=False,
           )),
        **dict((k, cfg.data[k]) for k in [
                   'seed',
                   'prefetch_num',
                   'pin_memory',
                   'persistent_workers',
               ] if k in cfg.data)
    }
    # step2: cfg.data.test_dataloader has higher priority
    test_loader_cfg = {
        **loader_cfg,
        **dict(shuffle=False, drop_last=False),
        **dict(workers_per_gpu=cfg.data.get('workers_per_gpu', 1)),
        **dict(samples_per_gpu=cfg.data.get('samples_per_gpu', 1)),
        **cfg.data.get('test_dataloader', {})
    }
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    model = build_posenet(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if not distributed:
        model = MMDataParallel(model, device_ids=[args.gpu_id])
        outputs = single_gpu_test(model, data_loader, return_heatmaps=True)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect, return_heatmaps=True)
        

    print("\nInference done")


    rank, _ = get_dist_info()
    eval_config = cfg.get('evaluation', {})
    eval_config = merge_configs(eval_config, dict(metric=args.eval))

    # print(outputs)
    # print(dir(dataset))
    # print(dataset.oks_thr)
    # print(dataset.use_gt_bbox)

    if rank == 0:

        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)

        config_name = ".".join(os.path.basename(args.config).split(".")[:-1])
        save_dir = os.path.join(
            cfg.data_root,
            # "test_all_visualization",
            "test_visualization",
            config_name,
        )
        # Prepare datastructure
        shutil.rmtree(save_dir, ignore_errors=True)
        os.makedirs(save_dir, exist_ok=True)
        
        heatmaps = np.concatenate([o["output_heatmap"] for o in outputs], axis=0)
        print("="*20)
        print("Heatmaps shape:", heatmaps.shape)
        print("Heatmaps avg:", np.mean(heatmaps))
        print("Heatmaps med:", np.median(heatmaps))
        print("Heatmaps min:", np.min(heatmaps))
        print("Heatmaps max:", np.max(heatmaps))
        plt.hist(heatmaps.flatten(), bins=100, log=True)
        plt.title("Heatmaps histogram (log scale)")
        plt.grid(True)
        plt.savefig(osp.join(save_dir, "test_heatmap_histogram.png"))
        plt.cla()

        print("+"*10)
        kpts_sums = np.sum(heatmaps, axis=(2, 3))
        print("Heatmaps avg sum:", np.mean(kpts_sums))
        print("Heatmaps med sum:", np.median(kpts_sums))
        print("Heatmaps min sum:", np.min(kpts_sums))
        print("Heatmaps max sum:", np.max(kpts_sums))
        plt.hist(kpts_sums.flatten(), bins=100)
        plt.title("Heatmaps sum histogram")
        plt.grid(True)
        plt.savefig(osp.join(save_dir, "test_heatmap_sum_histogram.png"))
        plt.cla()

        print("="*20)

        # print("="*30)
        # results_per_kpt = dataset.evaluate_per_kpts(outputs, cfg.work_dir, return_score=True, **eval_config)
        # kpts = dataset.coco.dataset["categories"][0]["keypoints"]
        # for i, result in enumerate(results_per_kpt):
        #     # print("=====", kpts[i], "="*30)
        #     print("{:s} -> {:.1f}".format(kpts[i], 100 * float(result['AP'])))
        # print("="*30)
        
        results, sorted_matches = dataset.evaluate(outputs, cfg.work_dir, return_score=True, **eval_config)
        print("Number of sorted matches:", len(sorted_matches))
        oks_list = np.array([m[2] for m in sorted_matches])
        print("Dataset evaluated")

        # Try to load dict with views
        views_dict_path = os.path.join(cfg.data_root, "annotations", "views.json")
        views_dict = None
        if os.path.exists(views_dict_path) and os.path.isfile(views_dict_path):
            views_dict = json.load(open(views_dict_path, "r"))

        # If views dict loaded, add OKS score to it
        if views_dict is not None:
            for img_name in views_dict.keys():
                views_dict[img_name]["oks_score"] = float(img_score_dict[img_name])
            
            # Save views dict
            views_dict_path = os.path.join(cfg.data_root, "annotations", "views_w_oks.json")
            print("Saving the views dict to {}".format(views_dict_path))
            json.dump(views_dict, open(views_dict_path, "w"), indent=2)
        # else:
        #     # Save OKS for each image for further eval
        #     coco_dict_with_oks = dataset.coco.dataset.copy()
            
        #     for i, ann in enumerate(coco_dict_with_oks["annotations"]):
        #         ann["oks"] = float(oks_list[i])
        #     coco_dict_with_oks_path = os.path.join(cfg.data_root, "annotations", "coco_dict_with_oks.json")
        #     print("Saving the coco with OKS dict to {}".format(coco_dict_with_oks_path))
        #     with open(coco_dict_with_oks_path, "w") as f:
        #         json.dump(coco_dict_with_oks, f, indent=2)

        # Save score histogram
        hist_oks_score = np.clip(oks_list, 0, 1)
        plt.hist(hist_oks_score, bins=100)
        plt.savefig(os.path.join(save_dir, "test_score_histogram.png"))
        plt.cla()

        if not hasattr(dataset, "coco"):
            ann_dict = json.load(open("/datagrid/personal/purkrmir/data/MPII/annotations/_mpii_trainval_custom.json", "r"))
       
        
        num_non_nan = (np.isnan(oks_list) == False).sum()
        print("There is {:d} non-NaN OKS scores out of {:d} samples".format(num_non_nan, len(oks_list)))

        draw_all = True
        if draw_all: 
            num_images = len(oks_list)
            indices_to_draw = list(range(num_images))
        else:
            num_images = 100
            print(1, len(oks_list), num_images)
            indices_to_draw = np.geomspace(1, len(oks_list), num=num_images) - 1
            indices_to_draw = np.unique(indices_to_draw.astype(int))

        # copy_worst = False
        # if copy_worst:
        #     print("Selecting 20% of hard negatives...")
        #     # Copy worst 20% of images
        #     worst_save_dir = os.path.join(
        #         cfg.data_root,
        #         "hard_negatives",
        #         config_name,
        #     )
        #     os.makedirs(worst_save_dir, exist_ok=True)
        #     num_images = int(num_non_nan * 0.2)
        #     worst_dataset = {
        #         "images": [],
        #         "annotations": [],
        #         "categories": dataset.coco.dataset["categories"],
        #     }
        #     for idx in tqdm(sorted_oks_per_sample[:num_images], ascii=True):
        #         annotation = dataset.coco.dataset["annotations"][idx]
        #         img_ann = dataset.coco.loadImgs(annotation["image_id"])[0]
        #         image_name = dataset.id2name[annotation["image_id"]]
        #         image_path = os.path.join(cfg.data_root, "val2017", image_name)
        #         shutil.copy(image_path, worst_save_dir)
        #         worst_dataset["annotations"].append(annotation)
        #         worst_dataset["images"].append(img_ann)
        #     with open(os.path.join(worst_save_dir, os.pardir, "{}_hard_negatives.json".format(config_name)), "w") as f:
        #         json.dump(worst_dataset, f, indent=2)            


        print("Drawing {:d} images ({:d} available)".format(len(indices_to_draw), len(oks_list)))
        for i in tqdm(indices_to_draw, ascii=True):
            dt, annotation, score = sorted_matches[i]
            oks_score_for_this_sample = score
            
            image_name = dataset.id2name[annotation["image_id"]]
            image_path = os.path.join(cfg.data_root, "val2017", image_name)
            
            pose_results = []
            gt_pose_results = []
            gt_pose_vis = []
            gt_pose_invis = []
            heatmaps = []
            
            kpt = np.array(annotation["keypoints"]).reshape(17, 3)
            visibility = kpt[:, -1]
            
            vis_kpt = kpt.copy()
            vis_kpt[visibility != 2, :] = 0
            
            invis_kpt = kpt.copy()
            invis_kpt[visibility != 1, :] = 0

            kpt[:, -1] = (kpt[:, -1] > 0).astype(int)
            invis_kpt[:, -1] = (invis_kpt[:, -1] > 0).astype(int)
            vis_kpt[:, -1] = (vis_kpt[:, -1] > 0).astype(int)
            
            bbox_wh = np.array(annotation["bbox"]).reshape(1, 4)
            gt_pose_results.append({
                "keypoints": kpt,
                "bbox": bbox_wh,
            })
            gt_pose_vis.append({
                "keypoints": vis_kpt,
                "bbox": bbox_wh,
            })
            gt_pose_invis.append({
                "keypoints": invis_kpt,
                "bbox": bbox_wh,
            })

            
            gt_img_id = annotation["image_id"]
            gt_id = annotation["id"]
            dt_img_id = dt["image_id"]
            bbox_cs = np.array(list(dt["center"]) + list(dt["scale"])).reshape((1, 4))
            bbox_wh = bbox_cs2xywh(bbox_cs[:, :2], bbox_cs[:, 2:], padding=1.0).reshape((1, 4))
            # bbox_xy = bbox_xywh2xyxy(bbox_wh).squeeze()
            # bbox_wh[bbox_wh < 0] = 0
            # bbox_xy[bbox_xy < 0] = 0
            
            pose_results.append({
                "keypoints": np.array(dt["keypoints"]).reshape(17, 3),
                "bbox": bbox_wh,
            })


            # print(output_list[idx]["image_paths"].replace("/", ""), image_path.replace("/", ""))
            assert gt_img_id == dt_img_id, "Image IDs does not equal, {:d} =!= {:d}".format(gt_img_id, dt_img_id)
            # assert output_list[idx]["image_paths"].replace("/", "") == image_path.replace("/", ""), "{:s} =!= {:s}".format(output_list[idx]["image_paths"], image_path)

            # heatmaps = np.array(heatmaps)
            # for joint_i in range(heatmaps.shape[1]):
            #     save_path = osp.join(save_dir, "{:04d}_vis_heatmap_{:02d}_{}".format(i, joint_i, image_name))
            #     joint_heatmap = (heatmaps[:, joint_i, :, :].squeeze()*255)
            #     joint_heatmap = np.clip(joint_heatmap, 0, 255)#.astype(np.uint8)
            #     joint_heatmap = cv2.resize(joint_heatmap, (np.array(joint_heatmap.shape) * 4).astype(int))
            #     mmcv.image.imwrite(joint_heatmap, save_path)
            #     print(save_path)
            #     print("Joint {:d}, min {:.2f}, max {:.2f}, conf {:.2f}".format(
            #         joint_i,
            #         np.min(joint_heatmap),
            #         np.max(joint_heatmap),
            #         pose_results[0]["keypoints"][joint_i, -1] * 255
            #     ))

            image_basename = ".".join(image_name.split(".")[:-1])
            image_ext = image_name.split(".")[-1]
            # save_path = osp.join(save_dir, "{:04d}_vis_{:s}-{:02d}.{}".format(0, image_basename, gt_id, "png"))
            save_path = osp.join(save_dir, "{:04d}_vis_{:s}-{:d}.{}".format(i, image_basename, gt_id, "png"))
            save_path_paper = osp.join(save_dir, "{:04d}_vis_{:s}-{:d}_paper.{}".format(i, image_basename, gt_id, "png"))
            dataset_info = DatasetInfo(cfg.data['test'].get('dataset_info', None))
            
            # Plot GT as GREEN 
            # dataset_info.pose_link_color = [[0, 255, 0] for _ in range(len(dataset_info.pose_link_color))]
            dataset_info.pose_kpt_color = [[0, 255, 0] for _ in range(len(dataset_info.pose_kpt_color))]
            
            # For the paper visualization, draw pose without bbox
            # save_img_paper = pose_visualization(
            #     image_path,
            #     pose_results,
            #     show_markers=False,
            #     line_type="solid",
            #     width_multiplier=2,
            #     show_bbox=False,
            # )

            # If GT or PRED kpts outside of the image, pad the image
            save_img = cv2.imread(image_path)
            min_x = 0
            min_y = 0
            max_x = save_img.shape[1]
            max_y = save_img.shape[0]
            for gt_pose in gt_pose_results:
                valid_kpts = gt_pose["keypoints"][:, -1] > 0
                if np.sum(valid_kpts) == 0:
                    continue
                min_x = min(min_x, int(np.min(gt_pose["keypoints"][valid_kpts, 0]))-10)
                min_y = min(min_y, int(np.min(gt_pose["keypoints"][valid_kpts, 1]))-10)
                max_x = max(max_x, int(np.max(gt_pose["keypoints"][valid_kpts, 0]))+10)
                max_y = max(max_y, int(np.max(gt_pose["keypoints"][valid_kpts, 1]))+10)
            for pose in pose_results:
                valid_kpts = pose["keypoints"][:, -1] > 0
                if np.sum(valid_kpts) == 0:
                    continue
                min_x = min(min_x, int(np.min(pose["keypoints"][valid_kpts, 0]))-10)
                min_y = min(min_y, int(np.min(pose["keypoints"][valid_kpts, 1]))-10)
                max_x = max(max_x, int(np.max(pose["keypoints"][valid_kpts, 0]))+10)
                max_y = max(max_y, int(np.max(pose["keypoints"][valid_kpts, 1]))+10)
            save_img = cv2.copyMakeBorder(
                save_img,
                -min_y,
                max_y - save_img.shape[0],
                -min_x,
                max_x - save_img.shape[1],
                cv2.BORDER_CONSTANT,
                value=[80, 80, 80],
            )

            # Shift kpts and bboxes by the padding
            for gt_pose in gt_pose_results:
                valid_kpts = gt_pose["keypoints"][:, -1] > 0
                if np.sum(valid_kpts) == 0:
                    continue
                gt_pose["keypoints"][valid_kpts, 0] -= min_x
                gt_pose["keypoints"][valid_kpts, 1] -= min_y
                gt_pose["bbox"][:, 0] -= min_x
                gt_pose["bbox"][:, 1] -= min_y
            for pose in pose_results:
                valid_kpts = pose["keypoints"][:, -1] > 0
                if np.sum(valid_kpts) == 0:
                    continue
                pose["keypoints"][valid_kpts, 0] -= min_x
                pose["keypoints"][valid_kpts, 1] -= min_y
                pose["bbox"][:, 0] -= min_x
                pose["bbox"][:, 1] -= min_y

            # print("-"*10, "\n", "-"*10)
            # print(oks_score_for_this_sample)
            # print(gt_pose_results)
            # print(pose_results)

            save_img = pose_visualization(
                save_img,
                gt_pose_results,
                show_markers=False,
                line_type="dashed",
                show_bbox=True,
            )

            # print(image_basename, gt_id)
            save_img = pose_visualization(
                save_img,
                pose_results,
                show_markers=True,
                line_type="solid",
                width_multiplier=1,
                show_bbox=True,
                conf_thr=0.3,
            )
            
            # Crop by the bbox
            # save_img = save_img[
            #     int(bbox_xy[1]):int(bbox_xy[3]),
            #     int(bbox_xy[0]):int(bbox_xy[2]), :]
            # save_img_paper = save_img_paper[
            #     int(bbox_xy[1]):int(bbox_xy[3]),
            #     int(bbox_xy[0]):int(bbox_xy[2]), :]
            
            save_img = cv2.putText(
                save_img,
                "{:.2f}".format(oks_score_for_this_sample),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 255, 0),
                thickness=2,
            )

            # if np.isnan(oks_score_for_this_sample):

            # print()
            # print("OKS score is NaN for image {:s}".format(image_name))
            # print(annotation)
            # print(pose_results)

            # If score is lower than 0.5, overlay the image with semi-transparent red color
            # alpha = 0.6
            # conf = 0.8
            # if oks_score_for_this_sample < conf:
            #     save_img_paper = cv2.addWeighted(
            #         save_img_paper,
            #         alpha,
            #         np.ones_like(save_img) * np.array([0, 0, 255], dtype=np.uint8),
            #         1 - alpha,
            #         0
            #     )

            try:
                mmcv.image.imwrite(save_img, save_path)
                # mmcv.image.imwrite(save_img_paper, save_path_paper)
            except:
                print("Failed to save image to {}".format(save_path))
                continue

        for k, v in sorted(results.items()):
            print(f'{k}: {v}')


if __name__ == '__main__':
    main()
