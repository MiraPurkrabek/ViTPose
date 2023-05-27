# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import shutil
import os.path as osp
import warnings
import numpy as np
import cv2

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

        results, wrong_images, oks_score, oks_sample_score = dataset.evaluate(outputs, cfg.work_dir, return_score=True, **eval_config)
        
        print("oks_score", oks_score.shape, np.mean(oks_score))
        print("oks_sample_score", oks_sample_score.shape, np.mean(oks_sample_score))
        print(np.isnan(oks_score).sum(), (oks_score <= 0).sum())
        print(np.isnan(oks_sample_score).sum(), (oks_sample_score <= 0).sum())

        valid_oks = oks_score[np.isnan(oks_score) == False]
        print("valid_oks", valid_oks.shape, valid_oks.mean())

        print(oks_score)
        print(oks_score.shape, len(dataset.coco.dataset["annotations"]))
        print(np.isnan(oks_score).sum(), (oks_score <= 0).sum(), (oks_score > 0).sum())

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

        # Try to load dict with views
        views_dict_path = os.path.join(cfg.data_root, "annotations", "views.json")
        views_dict = None
        if os.path.exists(views_dict_path) and os.path.isfile(views_dict_path):
            views_dict = json.load(open(views_dict_path, "r"))

        # If views dict loaded, add OKS score to it
        if views_dict is not None:
            for i, img_path in enumerate(wrong_images):
                img_name = os.path.basename(img_path)
                views_dict[img_name]["oks_score"] = oks_score[i]
            
            # Save views dict
            views_dict_path = os.path.join(cfg.data_root, "annotations", "views_w_oks.json")
            json.dump(views_dict, open(views_dict_path, "w"), indent=2)
        else:
            # Save OKS for each image for further eval
            coco_dict_with_oks = dataset.coco.dataset.copy()
            i = 0
            for img in coco_dict_with_oks["images"]:
                image_id = img["id"]
                for ann in coco_dict_with_oks["annotations"]:
                    if ann["image_id"] == image_id:
                        ann["oks"] = float(oks_sample_score[i])
                        i += 1
            
            # for i, ann in enumerate(coco_dict_with_oks["annotations"]):
            #     ann["oks"] = float(oks_sample_score[i])
            with open(os.path.join(cfg.data_root, "annotations", "coco_dict_with_oks.json"), "w") as f:
                json.dump(coco_dict_with_oks, f, indent=2)

        # Save score histogram
        hist_oks_score = oks_score[oks_score <= 1]
        hist_oks_score = hist_oks_score[hist_oks_score >= 0]
        plt.hist(hist_oks_score, bins=100)
        plt.savefig(os.path.join(save_dir, "test_score_histogram.png"))

        if not hasattr(dataset, "coco"):
            ann_dict = json.load(open("/datagrid/personal/purkrmir/data/MPII/annotations/_mpii_trainval_custom.json", "r"))

        num_images = 30
        indices_to_draw = np.geomspace(1, dataset.num_images, num=num_images) - 1
        indices_to_draw = np.unique(indices_to_draw.astype(int))

        # indices_to_draw = np.unique(list(range(dataset.num_images)))
        
        for i in indices_to_draw:
            image_path = wrong_images[i]
            image_name = osp.basename(image_path)
            pose_results = []
            gt_pose_results = []
            gt_pose_vis = []
            gt_pose_invis = []
            heatmaps = []
            if hasattr(dataset, "coco"):
                for ann in dataset.coco.dataset["annotations"]:
                    if ann["image_id"] == dataset.name2id[image_name]:
                        kpt = np.array(ann["keypoints"]).reshape(17, 3)
                        visibility = kpt[:, -1]
                        
                        vis_kpt = kpt.copy()
                        vis_kpt[visibility != 2, :] = 0
                        
                        invis_kpt = kpt.copy()
                        invis_kpt[visibility != 1, :] = 0

                        kpt[:, -1] = (kpt[:, -1] > 0).astype(int)
                        invis_kpt[:, -1] = (invis_kpt[:, -1] > 0).astype(int)
                        vis_kpt[:, -1] = (vis_kpt[:, -1] > 0).astype(int)
                        
                        bbox_wh = np.array(ann["bbox"]).reshape(1, 4)
                        gt_pose_results.append({
                            "keypoints": kpt,
                            "bbox": bbox_xywh2xyxy(bbox_wh),
                        })
                        gt_pose_vis.append({
                            "keypoints": vis_kpt,
                            "bbox": bbox_xywh2xyxy(bbox_wh),
                        })
                        gt_pose_invis.append({
                            "keypoints": invis_kpt,
                            "bbox": bbox_xywh2xyxy(bbox_wh),
                        })
            else:
                for gt in ann_dict[image_name]:
                    kpt = np.array(gt["joints"]).reshape(16, 2)
                    visibility = np.array(gt["joints_vis"]).reshape(16, 1)
                    kpt = np.concatenate([kpt, visibility], axis=1)
                    bbox_c = np.array(gt["center"])
                    bbox_s = np.array([gt["scale"], gt["scale"]])
                    bbox_wh = bbox_cs2xywh(bbox_c, bbox_s).reshape(1, 4)
                    gt_pose_results.append({
                        "keypoints": kpt,
                        "bbox": bbox_xywh2xyxy(bbox_wh)
                    })

            for batch in outputs:
                batch_images = np.array(batch["image_paths"])
                indices = np.where(image_path == batch_images)[0]

                if indices.size > 0:
                    for ind in indices:
                        bbox_cs = batch["boxes"][ind, :4].reshape((1, 4))
                        bbox_wh = bbox_cs2xywh(bbox_cs[:, :2], bbox_cs[:, 2:], padding=1.0).reshape((1, 4))
                        bbox_xy = bbox_xywh2xyxy(bbox_wh).squeeze()
                        
                        pose_results.append({
                            "keypoints": batch["preds"][ind, :, :],
                            "bbox": bbox_xy,
                        })

                        heatmap = (batch["output_heatmap"][ind, :, :, :]).squeeze()
                        heatmaps.append(heatmap)

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

            save_path = osp.join(save_dir, "{:04d}_vis_{}".format(i, image_name))
            dataset_info = DatasetInfo(cfg.data['test'].get('dataset_info', None))
            
            # Plot GT as GREEN 
            dataset_info.pose_link_color = [[0, 255, 0] for _ in range(len(dataset_info.pose_link_color))]
            dataset_info.pose_kpt_color = [[0, 255, 0] for _ in range(len(dataset_info.pose_kpt_color))]
            save_img = vis_pose_result(
                model,
                image_path,
                gt_pose_results,
                dataset=cfg.data['test']['type'],
                dataset_info=dataset_info,
                kpt_score_thr=0.3,
                radius=4,
                thickness=1,
                bbox_color="green",
                show=False,
                out_file=None
            )

            # Plot invisible GT as BLUE 
            dataset_info.pose_kpt_color = [[255, 0, 0] for _ in range(len(dataset_info.pose_kpt_color))]
            vis_pose_result(
                model,
                save_img,
                gt_pose_invis,
                dataset=cfg.data['test']['type'],
                dataset_info=dataset_info,
                kpt_score_thr=0.3,
                radius=4,
                thickness=1,
                bbox_color="green",
                show=False,
                out_file=save_path
            )

            # Plot PRED as RED
            dataset_info.pose_link_color = [[0, 0, 255] for _ in range(len(dataset_info.pose_link_color))]
            dataset_info.pose_kpt_color = [[0, 0, 255] for _ in range(len(dataset_info.pose_kpt_color))]
            save_img = vis_pose_result(
                model,
                save_path,
                pose_results,
                dataset=cfg.data['test']['type'],
                dataset_info=dataset_info,
                kpt_score_thr=cfg.data_cfg.vis_thr,
                radius=4,
                thickness=1,
                bbox_color="red",
                show=False,
                out_file=None,
            )
            save_img = cv2.putText(
                save_img,
                "{:.2f}".format(oks_score[i]),
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 255, 0),
                thickness=2,
            )
            mmcv.image.imwrite(save_img, save_path)


        for k, v in sorted(results.items()):
            print(f'{k}: {v}')


if __name__ == '__main__':
    main()
