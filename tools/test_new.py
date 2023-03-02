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
from mmpose.core.bbox.transforms import bbox_cs2xywh, bbox_xywh2xyxy

import json
import matplotlib.pyplot as plt

try:
    from mmcv.runner import wrap_fp16_model
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import wrap_fp16_model


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
        outputs = single_gpu_test(model, data_loader)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

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

        results, wrong_images, oks_score = dataset.evaluate(outputs, cfg.work_dir, return_score=True, **eval_config)

        # Save score histogram
        plt.hist(oks_score, bins=100)
        plt.savefig(os.path.join(cfg.work_dir, "test_score_histogram.png"))

        # Prepare datastructure
        shutil.rmtree(osp.join(cfg.work_dir, "vis"), ignore_errors=True)
        os.makedirs(osp.join(cfg.work_dir, "vis"), exist_ok=True)


        if not hasattr(dataset, "coco"):
            ann_dict = json.load(open("/datagrid/personal/purkrmir/data/MPII/annotations/_mpii_trainval_custom.json", "r"))

        num_images = 100
        indices_to_draw = np.geomspace(1, dataset.num_images, num=num_images) - 1
        indices_to_draw = np.unique(indices_to_draw.astype(int))
        
        print(oks_score[indices_to_draw])
        
        for i in indices_to_draw:
            image_path = wrong_images[i]
            image_name = osp.basename(image_path)
            pose_results = []
            gt_pose_results = []
            if hasattr(dataset, "coco"):
                for ann in dataset.coco.dataset["annotations"]:
                    if ann["image_id"] == dataset.name2id[image_name]:
                        kpt = np.array(ann["keypoints"])
                        kpt = kpt[np.mod(np.arange(51), 3) != 2].reshape(17, 2)
                        visibility = np.any(kpt > 0, axis=1).astype(float).reshape(17, 1)
                        kpt = np.concatenate([kpt, visibility], axis=1)
                        bbox_wh = np.array(ann["bbox"]).reshape(1, 4)
                        gt_pose_results.append({
                            "keypoints": kpt,
                            "bbox": bbox_xywh2xyxy(bbox_wh)
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

            save_path = osp.join(cfg.work_dir, "vis", "{:02d}_vis_{}".format(i, image_name))
            dataset_info = DatasetInfo(cfg.data['test'].get('dataset_info', None))
            
            # Plot GT as GREEN
            dataset_info.pose_link_color = [[0, 255, 0] for _ in range(len(dataset_info.pose_link_color))]
            dataset_info.pose_kpt_color = [[0, 255, 0] for _ in range(len(dataset_info.pose_kpt_color))]
            vis_pose_result(
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
                kpt_score_thr=0.3,
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
