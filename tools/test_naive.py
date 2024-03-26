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
from sklearn import metrics
from scipy import stats

from posevis import pose_visualization

try:
    from mmcv.runner import wrap_fp16_model
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import wrap_fp16_model


VISUALIZE = True
VIS_HEATMAPS = False
VISUALIZE_EVERY_TH = 100
DETAILED_PRINT = True

PROB_THR = 0.5


def print_and_visualize(
        arr,
        print_str="Distances",
        title="Plot title",
        fname="plot",
        show_vlines=True,
        save_dir=".",
        clip_values=True,
    ):
    print("-"*20)
    arr = np.array(arr)
    if DETAILED_PRINT:
        for kpt_i in range(arr.shape[1]):
            if np.isnan(arr[:, kpt_i]).all():
                continue
            arr_i = arr[:, kpt_i].flatten()
            arr_i = arr_i[~np.isnan(arr_i)]
            arr_mean = np.mean(arr_i)
            arr_med = np.median(arr_i)
            arr_std = np.std(arr_i)

            print("{:s} for {:14s} (min/med/avg/std/max):\t{:7.2f} / {:7.2f} / {:7.2f} / {:7.2f} / {:7.2f}".format(
                print_str,
                KEYPOINT_NAMES[kpt_i],
                np.min(arr_i),
                np.median(arr_i),
                np.mean(arr_i),
                np.std(arr_i),
                np.max(arr_i),
            ))
            if clip_values:
                arr_to_hist = arr_i.clip(
                    arr_mean - 3*arr_std,
                    arr_mean + 3*arr_std,
                )
            else:
                arr_to_hist = arr_i

            plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            counts, _, _ = plt.hist(arr_to_hist, bins=100, log=False)
            max_counts = np.array(counts).max()

            if show_vlines:
                plt.vlines(arr_mean, 0, max_counts, colors="r", linestyles="dashed", label="mean", linewidth=1)
                plt.vlines(arr_med, 0, max_counts, colors="g", linestyles="dashed", label="median", linewidth=1)
                if "relative" in title.lower():
                    plt.vlines(1.0, 0, max_counts, colors="black", linestyles="dashed", linewidth=1)
                
                plt.text(2.0*arr_mean, 1.0*max_counts, "{:.1e}".format(arr_mean), ha="right", va="top", color="r")
                plt.text(2.0*arr_med, 1.02*max_counts, "{:.1e}".format(arr_med), ha="right", va="top", color="g")
                plt.legend()
            plt.title("{:s} for {:14s}".format(title, KEYPOINT_NAMES[kpt_i]))
            plt.grid(True)
            plt.savefig(osp.join(save_dir, "{:s}_{:02d}.png".format(fname, kpt_i)))
            plt.clf()

    arr_i = arr.flatten()
    arr_i = arr_i[~np.isnan(arr_i)]
    arr_mean = np.mean(arr_i)
    arr_med = np.median(arr_i)
    arr_std = np.std(arr_i)

    print("{:s} for {:14s} (min/med/avg/std/max):\t{:7.2f} / {:7.2f} / {:7.2f} / {:7.2f} / {:7.2f}".format(
        print_str,
        "ALL",
        np.min(arr_i),
        np.median(arr_i),
        np.mean(arr_i),
        np.std(arr_i),
        np.max(arr_i),
    ))
    if clip_values:
        arr_to_hist = arr_i.clip(
            arr_mean - 3*arr_std,
            arr_mean + 3*arr_std,
        )
    else:
        arr_to_hist = arr_i

    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    counts, _, _ = plt.hist(arr_to_hist, bins=100, log=False)
    max_counts = np.array(counts).max()

    if show_vlines:
        plt.vlines(arr_mean, 0, max_counts, colors="r", linestyles="dashed", label="mean", linewidth=1)
        plt.vlines(arr_med, 0, max_counts, colors="g", linestyles="dashed", label="median", linewidth=1)
        if "relative" in title.lower():
            plt.vlines(1.0, 0, max_counts, colors="black", linestyles="dashed", linewidth=1)
        
        plt.text(2.0*arr_mean, 1.0*max_counts, "{:.1e}".format(arr_mean), ha="right", va="top", color="r")
        plt.text(2.0*arr_med, 1.02*max_counts, "{:.1e}".format(arr_med), ha="right", va="top", color="g")
        plt.legend()
    plt.title("{:s} for {:14s}".format(title, "ALL"))
    plt.grid(True)
    plt.savefig(osp.join(save_dir, "{:s}_{:s}.png".format(fname, "ALL")))
    plt.clf()


def plot_correlation_graph(
        x,
        y,
        title="Plot title",
        print_str="Correlation",
        fname="plot",
        save_dir=".",
        xlabel="x",
        ylabel="y",
        x_clip=None,
        y_clip=None,
        with_ROC=False,
        plot_cla=True,
):
    print("-"*20)
    x = np.array(x)
    y = np.array(y)
    assert x.shape == y.shape
    coeff_mean = []
    if DETAILED_PRINT:
        x = x.reshape(-1, 17)
        y = y.reshape(-1, 17)
        for kpt_i in range(x.shape[1]):
            
            if np.isnan(x[:, kpt_i]).all():
                continue
            if np.isnan(y[:, kpt_i]).all():
                continue

            xi = x[:, kpt_i].flatten()
            yi = y[:, kpt_i].flatten()

            nan_mask = np.logical_and(~np.isnan(xi), ~np.isnan(yi))
            if x_clip is not None:
                nan_mask = np.logical_and(nan_mask, xi >= x_clip[0])
                nan_mask = np.logical_and(nan_mask, xi <= x_clip[1])
            if y_clip is not None:
                nan_mask = np.logical_and(nan_mask, yi >= y_clip[0])
                nan_mask = np.logical_and(nan_mask, yi <= y_clip[1])

            xi = xi[nan_mask]
            yi = yi[nan_mask]

            # correlation_coeff = np.corrcoef(xi, yi)[0, 1]
            correlation_coeff, _ = stats.spearmanr(xi, yi)
            coeff_mean.append(correlation_coeff)
            print("{:s} for {:14s}:\t{:7.2f}".format(print_str, KEYPOINT_NAMES[kpt_i], correlation_coeff))

            plt.scatter(xi, yi, s=1.5, alpha=0.8)
            if with_ROC:
                roc_auc = metrics.roc_auc_score(yi, xi)
                plt.title("{:s} for {:14s} (corr {:.4f}, roc auc {:.4f})".format(title, KEYPOINT_NAMES[kpt_i], correlation_coeff, roc_auc))
            else:
                plt.yscale('log')
                plt.xscale('log')
                plt.title("{:s} for {:14s} (corr {:.4f})".format(title, KEYPOINT_NAMES[kpt_i], correlation_coeff))
            plt.grid(True)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.savefig(osp.join(save_dir, "{:s}_{:02d}.png".format(fname, kpt_i)))
            plt.clf()
    x = x.copy().flatten()
    y = y.copy().flatten()
    nan_mask = np.logical_and(~np.isnan(x), ~np.isnan(y))
    if x_clip is not None:
        nan_mask = np.logical_and(nan_mask, x >= x_clip[0])
        nan_mask = np.logical_and(nan_mask, x <= x_clip[1])
    if y_clip is not None:
        nan_mask = np.logical_and(nan_mask, y >= y_clip[0])
        nan_mask = np.logical_and(nan_mask, y <= y_clip[1])
    x = x[nan_mask]
    y = y[nan_mask]
    # correlation_coeff = np.corrcoef(x, y)[0, 1]
    correlation_coeff, _ = stats.spearmanr(x, y)
    if len(coeff_mean) > 0:
        coeff_mean = np.array(coeff_mean).mean()
    else:
        coeff_mean = np.nan
    
    # plt.scatter(x, y, s=1.5, alpha=0.8)
    if with_ROC:
        roc_auc = metrics.roc_auc_score(y, x)
        print("{:s} for {:14s}:\t{:7.2f} (mean {:.2f}; roc auc {:.2f})".format(print_str, "ALL", correlation_coeff, coeff_mean, roc_auc))
        # plt.title("{:s} for {:14s} (corr {:.4f}, roc auc {:.4f})".format(title, "ALL", correlation_coeff, roc_auc))
    else:
        # plt.yscale('log')
        # plt.xscale('log')
        print("{:s} for {:14s}:\t{:7.2f} (mean {:.2f})".format(print_str, "ALL", correlation_coeff, coeff_mean))
        # plt.title("{:s} for {:14s} (corr {:.4f})".format(title, "ALL", correlation_coeff))
    # plt.grid(True)
    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    # plt.savefig(osp.join(save_dir, "{:s}_ALL.png".format(fname)))
    # plt.clf()
    
    if with_ROC:
    
        # Plot ROC curve and save it
        fpr, tpr, _ = metrics.roc_curve(y, x)
        plt.plot(fpr,tpr,label="{:s}, auc={:.3f}".format(xlabel, roc_auc), linewidth=2.0)
        plt.legend()
        plt.grid(True)
        plt.title("ROC curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.savefig(osp.join(save_dir, "{:s}_ROC.png".format(fname)))
        if plot_cla:
            plt.clf()


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

def bbox_xywh2cs(bbox, aspect_ratio, padding=1., pixel_std=1.):
    """Transform the bbox format from (x,y,w,h) into (center, scale)

    Args:
        bbox (ndarray): Single bbox in (x, y, w, h)
        aspect_ratio (float): The expected bbox aspect ratio (w over h)
        padding (float): Bbox padding factor that will be multilied to scale.
            Default: 1.0
        pixel_std (float): The scale normalization factor. Default: 200.0

    Returns:
        tuple: A tuple containing center and scale.
        - np.ndarray[float32](2,): Center of the bbox (x, y).
        - np.ndarray[float32](2,): Scale of the bbox w & h.
    """

    x, y, w, h = bbox[:4]
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    scale = np.array([w, h], dtype=np.float32) / pixel_std
    scale = scale * padding

    return center, scale

KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


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

    assert args.n_gpus == 1, 'Multi-GPU inference is not supported due to batch mixing'

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

    # Only for HRNet model transition experiment
    # second_checkpoint = "work_dirs/ViTPose_small_coco_256x192_full_fromHTM_blackout_finetune/best_AP_epoch_87.pth"
    # load_checkpoint(model, second_checkpoint, map_location='cpu', strict=False)

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
        outputs, datas = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect, return_heatmaps=True,
                                 return_probs=True,
                                )
        

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
            cfg.VAL_COCO_ROOT,
            # "test_all_visualization",
            "test_visualization",
            config_name,
        )
        # Prepare datastructure
        if VISUALIZE:
            shutil.rmtree(save_dir, ignore_errors=True)
        os.makedirs(save_dir, exist_ok=True)
        
        heatmaps = np.concatenate([o["output_heatmap"] for o in outputs], axis=0)
        confidences = np.concatenate([o["preds"][:, :, -1].flatten() for o in outputs], axis=0)
        if 'output_probs' in outputs[0]:
            probs = np.concatenate([o["output_probs"] for o in outputs], axis=0)
        else:
            probs = np.zeros_like(confidences).reshape(-1, 17)
        if 'output_errors' in outputs[0]:
            errs = np.concatenate([o["output_errors"] for o in outputs], axis=0)
        else:
            errs = np.zeros_like(confidences).reshape(-1, 17)
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
        plt.clf()

        print("+"*10)
        kpts_sums = np.sum(heatmaps, axis=(2, 3))
        print("Heatmaps avg sum:", np.mean(kpts_sums))
        print("Heatmaps med sum:", np.median(kpts_sums))
        print("Heatmaps min sum:", np.min(kpts_sums))
        print("Heatmaps max sum:", np.max(kpts_sums))
        plt.hist(kpts_sums.flatten(), bins=100, log=True)
        plt.title("Heatmaps sum histogram (log scale)")
        plt.grid(True)
        plt.savefig(osp.join(save_dir, "test_heatmap_sum_histogram.png"))
        plt.clf()

        print("+"*10)
        print("Confidences avg:", np.mean(confidences))
        print("Confidences med:", np.median(confidences))
        print("Confidences min:", np.min(confidences))
        print("Confidences max:", np.max(confidences))
        plt.hist(confidences, bins=100, log=True)
        plt.title("Confidences histogram (log scale)")
        plt.grid(True)
        plt.savefig(osp.join(save_dir, "test_confidences_histogram.png"))
        plt.clf()

        print("+"*10)
        print("probs avg:", np.mean(probs))
        print("probs med:", np.median(probs))
        print("probs min:", np.min(probs))
        print("probs max:", np.max(probs))
        plt.hist(probs.flatten(), bins=100, log=False)
        plt.title("probs histogram (log scale)")
        plt.grid(True)
        plt.savefig(osp.join(save_dir, "test_probs_histogram.png"))
        plt.clf()

        print("+"*10)
        print("errs avg:", np.mean(errs))
        print("errs med:", np.median(errs))
        print("errs min:", np.min(errs))
        print("errs max:", np.max(errs))
        plt.hist(errs.flatten(), bins=100, log=True)
        plt.title("errs histogram (log scale)")
        plt.grid(True)
        plt.savefig(osp.join(save_dir, "test_errs_histogram.png"))
        plt.clf()

        print("="*20)

        # Replace confidence for error estimation
        for o in outputs:
            conf_errs = o["output_errors"]
            # conf_errs -= np.min(conf_errs)
            # conf_errs /= np.max(conf_errs)
            # conf_errs = 1 - conf_errs
            o["preds"][:, :, -1] = conf_errs.reshape(-1, 17)

        dataset.evaluate(outputs, cfg.work_dir, return_score=True, **eval_config)

        # return

        out_boxes = np.concatenate([o['boxes'] for o in outputs], axis=0)
        out_preds = np.concatenate([o['preds'] for o in outputs], axis=0)
        out_heatmaps = np.concatenate([o['output_heatmap'] for o in outputs], axis=0)
        metas = np.concatenate([d['img_metas'].data[0] for d in datas], axis=0)

        ious = []
        ious_per_kpt = []
        ious_per_vis = [[], []]
        dists_per_vis = [[], []]
        distances = []
        confidences = []
        kpt_in_am = []
        rel_distances = []
        ious_dict = {}
        outputs_dict = {}
        ious_per_num_kpts = {k: [] for k in range(18)}
        img_sources = os.path.join(cfg.VAL_COCO_ROOT, "val2017")
        size_plot_x = []
        size_plot_y = []
        i = -1
        n_images = len(metas)
        for pred, bbox, meta, htms, prob, err in tqdm(zip(out_preds, out_boxes, metas, out_heatmaps, probs, errs), total=n_images , ascii=True):
            i += 1
            pred_center = bbox[:2]
            pred_scale = bbox[2:4]
            pred_xywh = bbox_cs2xywh(pred_center, pred_scale, pixel_std=200)
            pred_center, pred_scale = bbox_xywh2cs(pred_xywh, pixel_std=200, aspect_ratio=3/4, padding=1.25)
            # print(prob, type(prob), prob.shape)
            # breakpoint()
            pred_i = {
                "keypoints": np.array(pred[:, :3]),
                "bbox": bbox_cs2xywh(pred_center, pred_scale, pixel_std=200),
                "prob": prob.squeeze().tolist(),
            }
            # pred_i["keypoints"][:, -1] = prob.squeeze()
            confidences.append(np.array(pred[:, 2]))
            gt_bbox = bbox_cs2xywh(meta["center"], meta["scale"], pixel_std=200)
            gt_vis = meta["joints_3d_visible"][:, 0].flatten()
            gt_v = gt_vis.copy()
            gt_vis[gt_vis > 2] = 0
            gt_i = {
                "keypoints": meta["orig_joints_3d"],
                "bbox": gt_bbox,
                "area": gt_bbox[2] * gt_bbox[3],
            }
            gt_i["keypoints"][:, -1] = gt_vis
            
            gt_v[gt_v == 0] = np.nan
            gt_v[gt_v == 1] = np.nan
            gt_v[gt_v == 2] = 1
            gt_v[gt_v == 3] = 0
            kpt_in_am.append(gt_v)
            # if not np.allclose(gt_i["bbox"], pred_i["bbox"]):
            #     print("Sample {:d}".format(i))
            #     print(gt_i)
            #     print(pred_i)
            #     print(gt_i["bbox"] - pred_i["bbox"])
            #     continue
            # assert np.allclose(gt_i["bbox"], pred_i["bbox"])
            oks, oks_per_kpt, dist_per_kpt = compute_oks(gt_i, pred_i)
            # if oks < 0.1:
            #     print("Sample {:d}".format(i))
            #     print(gt_i)
            #     print(pred_i)
            #     print(oks)
            #     breakpoint()
            ious.append(oks)
            ious_per_kpt.append(oks_per_kpt)
            distances.append(dist_per_kpt)

            # breakpoint()

            vis_1_oks = oks_per_kpt[gt_vis == 1]
            vis_2_oks = oks_per_kpt[gt_vis == 2]
            vis_1_dist = dist_per_kpt[gt_vis == 1]
            vis_2_dist = dist_per_kpt[gt_vis == 2]
            ious_per_vis[0].extend(vis_1_oks)
            ious_per_vis[1].extend(vis_2_oks)
            dists_per_vis[0].extend(vis_1_dist)
            dists_per_vis[1].extend(vis_2_dist)

            img_path = os.path.join(img_sources, meta["image_file"])
            fname, fext = os.path.basename(meta["image_file"]).split(".")
            save_path = os.path.join(
                save_dir,
                "{:06.4f}_{}_vis.{}".format(oks, fname, fext)
            )

            ann_id = 0
            key = "{}_{:d}".format(fname, ann_id)
            while key in ious_dict:
                ann_id += 1
                key = "{}_{:d}".format(fname, ann_id)
            ious_dict[key] = oks

            pred_i["keypoints"] = pred_i["keypoints"].flatten().tolist()
            pred_i["bbox"] = pred_i["bbox"].tolist()
            outputs_dict[key] = pred_i
            
            num_annotated_kpts = (gt_vis > 0).sum()
            ious_per_num_kpts[num_annotated_kpts].append(oks)
            size_plot_x.append(gt_bbox[2] * gt_bbox[3])
            size_plot_y.append(oks)

            ################################
            # # Visualize
            if VISUALIZE:
                # if np.all(prob >= 0):
                #     continue

                if not i % VISUALIZE_EVERY_TH == 0:
                    continue

                img = cv2.imread(img_path)

                img_cropped = img.copy()
                img_cropped = cv2.copyMakeBorder(
                    img_cropped,
                    int(0.25*img.shape[0]),
                    int(0.25*img.shape[0]),
                    int(0.25*img.shape[1]),
                    int(0.25*img.shape[1]),
                    cv2.BORDER_CONSTANT,
                    value=[0, 0, 0]
                )
                exbbox = gt_bbox.copy()
                exbbox[0] += int(0.25*img.shape[1])
                exbbox[1] += int(0.25*img.shape[0])
                img_cropped = img_cropped[
                    int(exbbox[1]):int(exbbox[1]+exbbox[3]),
                    int(exbbox[0]):int(exbbox[0]+exbbox[2]),
                ]

                if VIS_HEATMAPS:
                    for j_id in range(17):
                        img_j = img_cropped.copy()
                        if img_j.shape[0] <= 0 or img_j.shape[1] <= 0:
                            continue
                        htm_save_path = save_path.replace(
                            "{}_vis".format(fname),
                            "{}_htm_{:02d}".format(fname, j_id)
                        )
                        htm_j = htms[j_id]
                        if htm_j.sum() < 1:
                            htm_j *= 2 * np.pi * 4
                        H, W, _ = img_j.shape
                        if H < 256:
                            H = 256
                            W = int(256 * img_j.shape[1] / img_j.shape[0])
                        img_j = cv2.resize(img_j, (W, H))
                        htm_j = cv2.resize(htm_j, (W, H))
                        coarse_pred = np.array(np.unravel_index(htm_j.argmax(), htm_j.shape))
                        htm_j = cv2.applyColorMap((htm_j * 255).astype(np.uint8), cv2.COLORMAP_JET)
                        img_j = cv2.addWeighted(img_j, 0.3, htm_j, 0.7, 0)
                        img_j = cv2.drawMarker(
                            img_j,
                            tuple(coarse_pred[::-1]),
                            (255, 255, 255),
                            markerType=cv2.MARKER_CROSS,
                            markerSize=5,
                            thickness=1,
                        )
                        # Visualize probabilities
                        # p = 255 if prob[j_id] < 0 else 0
                        p = (1 - prob[j_id]) * 255
                        img_j = cv2.copyMakeBorder(img_j, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[p, p, p])

                        cv2.imwrite(htm_save_path, img_j)

                # If GT or PRED kpts outside of the image, pad the image
                min_x = 0
                min_y = 0
                max_x = img.shape[1]
                max_y = img.shape[0]
                for pose in [gt_i, pred_i]:
                    pose = np.array(pose["keypoints"]).reshape(17, 3)
                    valid_kpts = pose[:, -1] > 0
                    if np.sum(valid_kpts) == 0:
                        continue
                    min_x = min(min_x, int(np.min(pose[valid_kpts, 0]))-10)
                    min_y = min(min_y, int(np.min(pose[valid_kpts, 1]))-10)
                    max_x = max(max_x, int(np.max(pose[valid_kpts, 0]))+10)
                    max_y = max(max_y, int(np.max(pose[valid_kpts, 1]))+10)
                img = cv2.copyMakeBorder(
                    img,
                    -min_y,
                    max_y - img.shape[0],
                    -min_x,
                    max_x - img.shape[1],
                    cv2.BORDER_CONSTANT,
                    value=[80, 80, 80],
                )

                # Shift kpts and bboxes by the padding
                for pose in [gt_i, pred_i]:
                    ann = pose
                    pose = np.array(pose["keypoints"]).reshape(17, 3)
                    valid_kpts = pose[:, -1] > 0
                    if np.sum(valid_kpts) == 0:
                        continue
                    pose[valid_kpts, 0] -= min_x
                    pose[valid_kpts, 1] -= min_y
                    ann["bbox"][0] -= min_x
                    ann["bbox"][1] -= min_y

                    ann["keypoints"] = pose.flatten().tolist()


                img = pose_visualization(
                    img,
                    gt_i,
                    show_markers=False,
                    line_type="dashed",
                    show_bbox=True,
                )
                constant_errors = np.array([
                    0.010491,
                    0.010025,
                    0.010172,
                    0.012849,
                    0.013613,
                    0.024210,
                    0.024254,
                    0.023965,
                    0.025096,
                    0.027613,
                    0.027291,
                    0.034316,
                    0.035065,
                    0.027791,
                    0.026890,
                    0.029753,
                    0.029938,
                ])
                img = pose_visualization(
                    img,
                    pred_i,
                    show_markers=True,
                    line_type="dashed",
                    show_bbox=False,
                    width_multiplier=1.0,
                    # errors=constant_errors,
                    errors=dist_per_kpt,
                    conf_thr=PROB_THR,
                )
                vis_err = err.copy()
                vis_err[prob < PROB_THR] = 0
                img = pose_visualization(
                    img,
                    pred_i,
                    show_markers=True,
                    line_type="solid",
                    show_bbox=True,
                    width_multiplier=1.0,
                    errors=vis_err,
                    conf_thr=PROB_THR,
                )
                img = cv2.putText(
                    img,
                    "{:.2f}".format(oks),
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 255, 0),
                    thickness=2,
                )
                mmcv.image.imwrite(img, save_path)
        kpt_in_am = np.array(kpt_in_am)
        
        print("-"*20)

        ious = np.array(ious)
        print("OKS avg: {:.4f}".format(np.mean(ious)))
        print("OKS med: {:.4f}".format(np.median(ious)))
        print("-"*20)

        # for k, v in ious_per_num_kpts.items():
        #     ious_per_num_kpts[k] = np.array(v)
        #     if len(v) == 0:
        #         continue
        #     print("OKS per num_kpts {:2d} (min, med, avg, max):\t{:.2f} / {:.2f} / {:.2f} / {:.2f}\t({:d})".format(
        #         k,
        #         np.min(ious_per_num_kpts[k]),
        #         np.median(ious_per_num_kpts[k]),
        #         np.mean(ious_per_num_kpts[k]),
        #         np.max(ious_per_num_kpts[k]),
        #         len(ious_per_num_kpts[k]),
        #     ))

        # # Print and visualize 'distances' (= measured error between predictions and GT)
        # print_and_visualize(
        #     distances*100,
        #     print_str="GT-PRED L2 distances",
        #     title="GT-PRED L2 distances",
        #     fname="test_err_histogram",
        #     show_vlines=True,
        #     save_dir=save_dir,
        # )

        # # Print and visualize 'ious' (= OKS between predictions and GT)
        # print_and_visualize(
        #     ious_per_kpt,
        #     print_str="OKS",
        #     title="OKS",
        #     fname="test_oks_histogram",
        #     show_vlines=False,
        #     save_dir=save_dir,
        #     clip_values=False,
        # )
        
        # Print and visualize 'adjusted errors' (measured error between predictions and GT
        # adjusted for the mean error of each keypoint type)
        adj_distances = np.array(distances)
        # adj_distances[kpt_in_am <= 0] = np.nan
        # adj_distances[probs < 0.5] = np.nan
        # print("+"*20)
        constant_errors = []
        for kpt_i in range(adj_distances.shape[1]):
            if np.isnan(adj_distances[:, kpt_i]).all():
                continue
            arr = adj_distances[:, kpt_i].flatten()
            arr_mean = np.median(arr[~np.isnan(arr)])
            constant_errors.append(arr_mean)
            arr = arr - arr_mean
            adj_distances[:, kpt_i] = arr
            # print("Adjusted error for {:14s} (mean): {:f}".format(
            #     KEYPOINT_NAMES[kpt_i],
            #     arr_mean,
            # ))
        constant_errors = np.array(constant_errors)

        print_and_visualize(
            adj_distances*100,
            print_str="Adjusted L2 distances",
            title="Adjusted L2 distances",
            fname="test_adj_err_histogram",
            show_vlines=True,
            save_dir=save_dir,
        )

        # Print and visualize 'better or equal' error
        print("-"*20)
        base_useful_distances = (distances - np.array(constant_errors))
        # print_and_visualize(
        #     base_useful_distances,
        #     print_str="Adjusted useful distances",
        #     title="'Useful' distances abs(errs - distances)",
        #     fname="test_adj_usf_err_histogram",
        #     show_vlines=True,
        #     save_dir=save_dir,
        # )   
        base_useful_distances = base_useful_distances.flatten()
        dist_to_norm = np.array(distances).flatten()[~np.isnan(base_useful_distances)]
        base_useful_distances = base_useful_distances[~np.isnan(base_useful_distances)]
        base_mae = np.mean(np.abs(base_useful_distances))
        base_medae = np.median(np.abs(base_useful_distances))
        base_mape = np.mean(np.abs(base_useful_distances / dist_to_norm))
        base_medape = np.median(np.abs(base_useful_distances / dist_to_norm))
        # base_useful_distances = np.clip(base_useful_distances, 0, base_useful_distances.max()) * 100
        print("{:.2f}% of keypoints have better or equal than predicted error".format(
            100 * (base_useful_distances <= 0).sum() / base_useful_distances.size,
        ))
        print("MAE   : {:.2f}".format(base_mae*100))
        print("MedAE : {:.2f}".format(base_medae*100))
        print("MAPE  : {:.2f}".format(base_mape*100))
        print("MedAPE: {:.2f}".format(base_medape*100))


        # Print and visualize 'absolute distances' (distances - errs)
        abs_distances = np.array(distances - errs) * 100
        # abs_distances[kpt_in_am <= 0] = np.nan
        # abs_distances[probs < 0.5] = np.nan
        print_and_visualize(
            abs_distances,
            print_str="Absolute distances",
            title="Absolute distances (dist - predErr)",
            fname="test_abs_err_histogram",
            show_vlines=True,
            save_dir=save_dir,
        )
        abs_distances = abs_distances.flatten()
        dist_to_norm = np.array(distances).flatten()[~np.isnan(abs_distances)]
        abs_distances = abs_distances[~np.isnan(abs_distances)]
        mae = np.mean(np.abs(abs_distances))
        medae = np.median(np.abs(abs_distances))
        mape = np.mean(np.abs(abs_distances / dist_to_norm))
        medape = np.median(np.abs(abs_distances))
        # breakpoint()    
        print("{:.2f}% of keypoints have better or equal than predicted error".format(
            100 * (abs_distances <= 0).sum() / abs_distances.size,
        ))
        print("MAE   : {:.2f}".format(mae*100))
        print("MedAE : {:.2f}".format(medae*100))
        print("MAPE  : {:.2f}".format(mape*100))
        print("MedAPE: {:.2f}".format(medape*100))

        # Print and visualize 'relative distances' (errs / distances)
        # rel_distances = np.array(errs / distances)
        # print_and_visualize(
        #     rel_distances,
        #     print_str="Relative distances",
        #     title="Relative distances (predErr / dist)",
        #     fname="test_rel_err_histogram",
        #     show_vlines=True,
        #     save_dir=save_dir,
        # )

        print("-"*20)
        for v in range(2):
            arr = np.array(ious_per_vis[v])
            if np.isnan(arr).all():
                continue
            print("OKS for vis {:d} (min, med, avg, max):\t{:.2f} / {:.2f} / {:.4f} / {:.2f}".format(
                v+1,
                np.nanmin(arr),
                np.nanmedian(arr),
                np.nanmean(arr),
                np.nanmax(arr),
            ))


        # Plot correlation between predicted and GT errors
        plot_correlation_graph(
            distances,
            errs,
            title="e* vs. e^ correlation",
            fname="test_Estar_vs_E_corr",
            save_dir=save_dir,
            print_str="Correlation between e* and e^",
            xlabel="e*",
            ylabel="e^",
        )
        
        # Plot correlation between confidences and GT errors
        plot_correlation_graph(
            distances,
            1- np.array(confidences),
            title="e* vs. conf correlation",
            fname="test_Estar_vs_conf",
            save_dir=save_dir,
            print_str="Correlation between e* and conf",
            xlabel="e*",
            ylabel="1-conf",
            # y_clip=(0, 0.8),
        )
        
        # Plot correlation between confidences and GT errors
        # plot_correlation_graph(
        #     distances,
        #     probs * errs,
        #     title="e* vs. prob*err correlation",
        #     fname="test_Estar_vs_probErr",
        #     save_dir=save_dir,
        #     print_str="Correlation between e* and prob*err",
        #     xlabel="e*",
        #     ylabel="prob * err",
        #     # y_clip=(0, 0.8),
        # )

        # Plot correlation between confidences and predicted errors
        # plot_correlation_graph(
        #     confidences,
        #     errs,
        #     title="conf vs. e^ correlation",
        #     fname="test_E_vs_conf_corr",
        #     save_dir=save_dir,
        #     print_str="Correlation between conf and e^",
        #     xlabel="conf",
        #     ylabel="e^",
        #     # x_clip=(0, 0.8),
        # )

        kpt_in_am = np.array(kpt_in_am)
        if len(np.unique(kpt_in_am[~np.isnan(kpt_in_am)])) > 1:
            plot_correlation_graph(
                confidences,
                kpt_in_am,
                title="conf vs. in AM correlation",
                fname="test_conf_vs_iAM_corr",
                save_dir=save_dir,
                print_str="Correlation between conf and in AM",
                xlabel="conf",
                ylabel="in AM",
                with_ROC=len(np.unique(kpt_in_am[~np.isnan(kpt_in_am)])) > 1,
                plot_cla=False,
            )

            plot_correlation_graph(
                probs,
                kpt_in_am,
                title="prob vs. in AM correlation",
                fname="test_prob_vs_iAM_corr",
                save_dir=save_dir,
                print_str="Correlation between prob and in AM",
                xlabel="prob",
                ylabel="in AM",
                with_ROC=len(np.unique(kpt_in_am[~np.isnan(kpt_in_am)])) > 1,
            )

        # Compute number of keypoints, that have much lower e^ than e* and has high confidence/probability
        print("-"*20)
        rel_distances = np.array(errs) / np.array(distances)
        rel_dist_mask = rel_distances < (1/1.5)
        conf_dist_mask = np.array(confidences) > 0.3
        prob_dist_mask = np.array(probs) > 0.3
        
        non_nan_mask = ~np.isnan(rel_distances)
        non_nan_mask = non_nan_mask & (~np.isnan(confidences))
        non_nan_mask = non_nan_mask & (~np.isnan(probs))

        rel_dist_mask = rel_dist_mask & non_nan_mask
        conf_dist_mask = conf_dist_mask & non_nan_mask
        prob_dist_mask = prob_dist_mask & non_nan_mask

        num_kpts = (~np.isnan(rel_distances)).sum()
        print("Number of keypoints with rel_dist < 0.5: {:d} ({:.4f})".format(
            np.sum(rel_dist_mask),
            np.sum(rel_dist_mask) / num_kpts,
        ))
        print("Number of keypoints with conf > 0.3: {:d} ({:.4f})".format(
            np.sum(conf_dist_mask),
            np.sum(conf_dist_mask) / num_kpts,
        ))
        print("Number of keypoints with prob > 0.3: {:d} ({:.4f})".format(
            np.sum(prob_dist_mask),
            np.sum(prob_dist_mask) / num_kpts,
        ))
        print("Number of keypoints with rel_dist < 0.5 and conf > 0.3: {:d} ({:.4f})".format(
            np.sum(rel_dist_mask & conf_dist_mask),
            np.sum(rel_dist_mask & conf_dist_mask) / num_kpts,
        ))
        print("Number of keypoints with rel_dist < 0.5 and prob > 0.3: {:d} ({:.4f})".format(
            np.sum(rel_dist_mask & prob_dist_mask),
            np.sum(rel_dist_mask & prob_dist_mask) / num_kpts,
        ))
        print("Number of keypoints with rel_dist < 0.5 and conf > 0.3 and prob > 0.3: {:d} ({:.4f})".format(
            np.sum(rel_dist_mask & conf_dist_mask & prob_dist_mask),
            np.sum(rel_dist_mask & conf_dist_mask & prob_dist_mask) / num_kpts,
        ))
            


        with open(osp.join(save_dir, "test_oks.json"), "w") as f:
            json.dump(ious_dict, f, indent=2)
        with open(osp.join(save_dir, "test_results.json"), "w") as f:
            json.dump(outputs_dict, f, indent=2)
        
        plt.hist(ious, bins=100, log=True)
        plt.title("OKS histogram (log scale)")
        plt.grid(True)
        plt.savefig(osp.join(save_dir, "test_oks_histogram.png"))
        plt.clf()

        plt.plot(size_plot_x, size_plot_y, "o", markersize=1)
        plt.title("OKS vs bbox size")
        plt.grid(True)
        plt.xlabel("bbox size")
        plt.ylabel("OKS")
        plt.savefig(osp.join(save_dir, "test_oks_bbox_size.png"))
        plt.clf()


def compute_oks(gt, dt, use_area=True):
    sigmas = np.array(
                [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89])/10.0
    vars = (sigmas * 2)**2
    k = len(sigmas)
    visibility_condition = lambda x: x > 0
    g = np.array(gt['keypoints']).reshape(k, 3)
    xg = g[:, 0]; yg = g[:, 1]; vg = g[:, 2]
    k1 = np.count_nonzero(visibility_condition(vg))
    bb = gt['bbox']
    x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
    y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
    
    d = np.array(dt['keypoints']).reshape((k, 3))
    xd = d[:, 0]; yd = d[:, 1]
            
    if k1>0:
        # measure the per-keypoint distance if keypoints visible
        dx = xd - xg
        dy = yd - yg

    else:
        # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
        z = np.zeros((k))
        dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
        dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
        # dx = np.ones(dx.shape) * 100
        # dy = np.ones(dy.shape) * 100

    if use_area:
        e = (dx**2 + dy**2) / vars / (gt['area']+np.spacing(1)) / 2
        # breakpoint()
        # print("area", gt["area"])
    else:
        tmparea = gt['bbox'][3] * gt['bbox'][2] * 0.53
        # print("tmparea", tmparea)
        e = (dx**2 + dy**2) / vars / (tmparea+np.spacing(1)) / 2
    
    center, scale = bbox_xywh2cs(gt['bbox'], 3/4, pixel_std=200.0, padding=1.25)
    ex_bbox = bbox_cs2xywh(center, scale, pixel_std=200.0)
    norm_area = np.sqrt(ex_bbox[2]**2 + ex_bbox[3]**2)
    # heatmap_area = np.sqrt(64 * 48)
    heatmap_area = 1.0
    normalized_dist = np.sqrt((dx**2 + dy**2)) / norm_area * heatmap_area
    
    oks_per_kpt = np.exp(-e)
    dist_per_kpt = np.ones_like(oks_per_kpt) * np.nan
    if k1 > 0:
        dist_per_kpt[visibility_condition(vg)] = normalized_dist[visibility_condition(vg)]
        e=e[visibility_condition(vg)]
        # e=e[vg > 0]
        # e=e[(vg > 0) & (det_conf > 0.3)]
        oks_per_kpt[~visibility_condition(vg)] = np.nan
    oks = np.sum(np.exp(-e)) / e.shape[0]

    # if oks < 0.1:
    #     breakpoint()
    # print()
    # print(oks)
    return oks, oks_per_kpt, dist_per_kpt

if __name__ == '__main__':
    main()
