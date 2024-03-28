# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict, defaultdict

import json_tricks as json
import numpy as np
from mmcv import Config, deprecated_api_warning

from sklearn import metrics

from xtcocotools.cocoeval import COCOeval as _COCOeval
from ._cocoeval import COCOeval

from copy import deepcopy


from ....core.post_processing import oks_nms, soft_oks_nms
from ...builder import DATASETS
from ..base import Kpt2dSviewRgbImgTopDownDataset

import warnings


@DATASETS.register_module()
class TopDownCocoDataset(Kpt2dSviewRgbImgTopDownDataset):
    """CocoDataset dataset for top-down pose estimation.

    "Microsoft COCO: Common Objects in Context", ECCV'2014.
    More details can be found in the `paper
    <https://arxiv.org/abs/1405.0312>`__ .

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    COCO keypoint indexes::

        0: 'nose',
        1: 'left_eye',
        2: 'right_eye',
        3: 'left_ear',
        4: 'right_ear',
        5: 'left_shoulder',
        6: 'right_shoulder',
        7: 'left_elbow',
        8: 'right_elbow',
        9: 'left_wrist',
        10: 'right_wrist',
        11: 'left_hip',
        12: 'right_hip',
        13: 'left_knee',
        14: 'right_knee',
        15: 'left_ankle',
        16: 'right_ankle'

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_info (DatasetInfo): A class containing all dataset info.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 test_mode=False):

        if dataset_info is None:
            warnings.warn(
                'dataset_info is missing. '
                'Check https://github.com/open-mmlab/mmpose/pull/663 '
                'for details.', DeprecationWarning)
            cfg = Config.fromfile('configs/_base_/datasets/coco.py')
            dataset_info = cfg._cfg_dict['dataset_info']

        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode)

        self.use_gt_bbox = data_cfg['use_gt_bbox']
        self.bbox_file = data_cfg['bbox_file']
        self.det_bbox_thr = data_cfg.get('det_bbox_thr', 0.0)
        self.use_nms = data_cfg.get('use_nms', True)
        self.soft_nms = data_cfg['soft_nms']
        self.nms_thr = data_cfg['nms_thr']
        self.oks_thr = data_cfg['oks_thr']
        self.vis_thr = data_cfg['vis_thr']

        self.db = self._get_db()

        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

    def _get_db(self):
        """Load dataset."""
        if (not self.test_mode) or self.use_gt_bbox:
            # use ground truth bbox
            gt_db = self._load_coco_keypoint_annotations()
        else:
            # use bbox from detection
            gt_db = self._load_coco_person_detection_results()
        return gt_db

    def _load_coco_keypoint_annotations(self):
        """Ground truth bbox and keypoints."""
        gt_db = []
        for img_id in self.img_ids:
            gt_db.extend(self._load_coco_keypoint_annotation_kernel(img_id))
        return gt_db

    def _load_coco_keypoint_annotation_kernel(self, img_id):
        """load annotation from COCOAPI.

        Note:
            bbox:[x1, y1, w, h]

        Args:
            img_id: coco image id

        Returns:
            dict: db entry
        """
        try:
            img_ann = self.coco.loadImgs(img_id)[0]
        except TypeError:
            img_ann = self.coco.loadImgs([img_id])[0]
        width = img_ann['width']
        height = img_ann['height']
        num_joints = self.ann_info['num_joints']

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        objs = self.coco.loadAnns(ann_ids)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            if 'bbox' not in obj:
                continue
            x, y, w, h = obj['bbox']
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width - 1, x1 + max(0, w - 1))
            y2 = min(height - 1, y1 + max(0, h - 1))
            if ('area' not in obj or obj['area'] > 0) and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                valid_objs.append(obj)
        objs = valid_objs

        bbox_id = 0
        rec = []
        for obj in objs:
            if 'keypoints' not in obj:
                continue
            if max(obj['keypoints']) == 0:
                continue
            if 'num_keypoints' in obj and obj['num_keypoints'] == 0:
                continue
            joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)

            keypoints = np.array(obj['keypoints']).reshape(-1, 3)
            joints_3d[:, :2] = keypoints[:, :2]
            # joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3])
            joints_3d_visible[:, :2] = keypoints[:, 2:3].astype(int)

            center, scale = self._xywh2cs(*obj['clean_bbox'][:4])

            image_file = osp.join(self.img_prefix, self.id2name[img_id])
            rec.append({
                'image_file': image_file,
                'center': center,
                'scale': scale,
                'bbox': obj['clean_bbox'][:4],
                'rotation': 0,
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'dataset': self.dataset_name,
                'bbox_score': 1,
                'bbox_id': bbox_id
            })
            bbox_id = bbox_id + 1

        return rec

    def _load_coco_person_detection_results(self):
        """Load coco person detection results."""
        num_joints = self.ann_info['num_joints']
        all_boxes = None
        with open(self.bbox_file, 'r') as f:
            all_boxes = json.load(f)

        if not all_boxes:
            raise ValueError('=> Load %s fail!' % self.bbox_file)

        print(f'=> Total boxes: {len(all_boxes)}')

        kpt_db = []
        bbox_id = 0
        for det_res in all_boxes:
            if det_res['category_id'] != 1:
                continue

            image_file = osp.join(self.img_prefix,
                                  self.id2name[det_res['image_id']])
            box = det_res['bbox']
            score = det_res['score']

            if score < self.det_bbox_thr:
                continue

            center, scale = self._xywh2cs(*box[:4])
            joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d_visible = np.ones((num_joints, 3), dtype=np.float32)
            kpt_db.append({
                'image_file': image_file,
                'center': center,
                'scale': scale,
                'rotation': 0,
                'bbox': box[:4],
                'bbox_score': score,
                'dataset': self.dataset_name,
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'bbox_id': bbox_id
            })
            bbox_id = bbox_id + 1
        print(f'=> Total boxes after filter '
              f'low score@{self.det_bbox_thr}: {bbox_id}')
        return kpt_db

    @deprecated_api_warning(name_dict=dict(outputs='results'))
    def evaluate(self, results, res_folder=None, metric='mAP', return_score=False, **kwargs):
        """Evaluate coco keypoint results. The pose prediction results will be
        saved in ``${res_folder}/result_keypoints.json``.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmap height: H
            - heatmap width: W

        Args:
            results (list[dict]): Testing results containing the following
                items:

                - preds (np.ndarray[N,K,3]): The first two dimensions are \
                    coordinates, score is the third dimension of the array.
                - boxes (np.ndarray[N,6]): [center[0], center[1], scale[0], \
                    scale[1],area, score]
                - image_paths (list[str]): For example, ['data/coco/val2017\
                    /000000393226.jpg']
                - heatmap (np.ndarray[N, K, H, W]): model output heatmap
                - bbox_id (list(int)).
            res_folder (str, optional): The folder to save the testing
                results. If not specified, a temp folder will be created.
                Default: None.
            metric (str | list[str]): Metric to be performed. Defaults: 'mAP'.

        Returns:
            dict: Evaluation results for evaluation metric.
        """
        if isinstance(results, tuple):
            results = results[0]

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['mAP']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        if res_folder is not None:
            tmp_folder = None
            res_file = osp.join(res_folder, 'result_keypoints.json')
        else:
            tmp_folder = tempfile.TemporaryDirectory()
            res_file = osp.join(tmp_folder.name, 'result_keypoints.json')

        kpts = defaultdict(list)

        # breakpoint()
        for result in results:
            preds = result['preds']
            boxes = result['boxes']
            image_paths = result['image_paths']
            bbox_ids = result['bbox_ids']
            if "output_probs" in result:
                probs = result['output_probs']
            else:
                probs = np.ones_like(bbox_ids)

            batch_size = len(image_paths)
            for i in range(batch_size):
                image_id = self.name2id[image_paths[i][len(self.img_prefix):]]
                kpts[image_id].append({
                    'keypoints': preds[i],
                    'center': boxes[i][0:2],
                    'scale': boxes[i][2:4],
                    'area': boxes[i][4],
                    'score': boxes[i][5],
                    'image_id': image_id,
                    'bbox_id': bbox_ids[i],
                    'prob': probs[i]
                })
        kpts = self._sort_and_unique_bboxes(kpts)

        # rescoring and oks nms
        num_joints = self.ann_info['num_joints']
        vis_thr = self.vis_thr
        oks_thr = self.oks_thr
        valid_kpts = []
        for image_id in kpts.keys():
            img_kpts = kpts[image_id]
            for n_p in img_kpts:
                box_score = n_p['score']
                kpt_score = 0
                valid_num = 0
                for n_jt in range(0, num_joints):
                    t_s = n_p['keypoints'][n_jt][2]
                    if t_s > vis_thr:
                        kpt_score = kpt_score + t_s
                        valid_num = valid_num + 1
                if valid_num != 0:
                    kpt_score = kpt_score / valid_num
                # rescoring
                n_p['score'] = kpt_score * box_score

            if self.use_nms:
                nms = soft_oks_nms if self.soft_nms else oks_nms
                keep = nms(img_kpts, oks_thr, sigmas=self.sigmas)
                valid_kpts.append([img_kpts[_keep] for _keep in keep])
            else:
                valid_kpts.append(img_kpts)

        # breakpoint()

        self._write_coco_keypoint_results(valid_kpts, res_file)

        # do evaluation only if the ground truth keypoint annotations exist
        if 'annotations' in self.coco.dataset:
            if return_score:
                info_str, sorted_matches, sort_idx = self._do_python_keypoint_eval(res_file, return_wrong_images=True, kpts=valid_kpts)
            else:
                info_str = self._do_python_keypoint_eval(res_file, return_wrong_images=False, kpts=valid_kpts)
            name_value = OrderedDict(info_str)

        if tmp_folder is not None:
            tmp_folder.cleanup()

        if return_score:
            return name_value, sorted_matches, sort_idx
        else:
            return name_value

    def _write_coco_keypoint_results(self, keypoints, res_file):
        """Write results into a json file."""
        data_pack = [{
            'cat_id': self._class_to_coco_ind[cls],
            'cls_ind': cls_ind,
            'cls': cls,
            'ann_type': 'keypoints',
            'keypoints': keypoints
        } for cls_ind, cls in enumerate(self.classes)
                     if not cls == '__background__']

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])

        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        """Get coco keypoint results."""
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array(
                [img_kpt['keypoints'] for img_kpt in img_kpts])
            key_points = _key_points.reshape(-1,
                                             self.ann_info['num_joints'] * 3)

            result = [{
                'image_id': img_kpt['image_id'],
                'category_id': cat_id,
                'keypoints': key_point.tolist(),
                'score': float(img_kpt['score']),
                'center': img_kpt['center'].tolist(),
                'scale': img_kpt['scale'].tolist()
            } for img_kpt, key_point in zip(img_kpts, key_points)]

            cat_results.extend(result)

        return cat_results
    

    def _eval_keypoints_classification(
            self, gt, dt, zeros=[3], ones=[1, 2], nan=[0], prefix="", balanced=True, verbose=True
        ):
        # Copy to keep the original data
        gt = gt.copy()
        dt = dt.copy()
        
        info_str = []

        # Filter the gt
        for n in nan:
            gt[gt == n] = np.nan
        for z in zeros:
            gt[gt == z] = 0
        for o in ones:
            gt[gt == o] = 1
        gt_mask = np.isnan(gt)
        gt = gt[~gt_mask].astype(int)
        dt = dt[~gt_mask]

        unique_gt = np.unique(gt)
        if len(unique_gt) != 2:
            warnings.warn("There is not 2 unique values in gt", RuntimeWarning)
            return info_str
        
        if balanced:
            # Take the same amount of in and out kpts
            n_kpts = min(len(gt[gt == 0]), len(gt[gt == 1]))
            inds_0 = np.random.choice(np.where(gt == 0)[0], n_kpts, replace=False)
            inds_1 = np.random.choice(np.where(gt == 1)[0], n_kpts, replace=False)
            gt = np.concatenate([gt[inds_0], gt[inds_1]])
            dt = np.concatenate([dt[inds_0], dt[inds_1]])

        if verbose:
            print("{:s}: There is {} kpts for eval, {} are '0' and {} are '1'".format(
                prefix.upper(),
                len(gt), len(gt[gt == 0]), len(gt[gt == 1]) 
            ))
        
        roc_auc = metrics.roc_auc_score(gt, dt)
        info_str = [(f'{prefix}auc', roc_auc)]

        thresholds = np.arange(0, 1, 0.05)
        accuracies = np.zeros(len(thresholds))
        for i, thr in enumerate(thresholds):
            accuracies[i] = metrics.accuracy_score(gt, dt > thr)
        
        best_i = np.argmax(accuracies)
        best_acc = accuracies[best_i]
        best_thr = thresholds[best_i]

        if verbose:
            print("{:s}: The best accuracy is {:.2f}% with threshold {:.2f}".format(
                prefix.upper(),
                best_acc*100, best_thr
            ))

        info_str.append((f'{prefix}best_acc', best_acc))
        info_str.append((f'{prefix}best_thr', best_thr))

        return info_str
    

    def _do_python_keypoint_eval(self, res_file, return_wrong_images=False, kpts=None):
        """Keypoint evaluation using COCOAPI."""
        coco_det = self.coco.loadRes(res_file)

        info_str = []
        best_prob_thr = 0.5
        best_conf_thr = 0.5

        if len(self.coco.imgs) == len(self.coco.anns):
            sorted_ids = []
            gt_probs = []
            pred_probs = []
            pred_confs = []
            for img_k in kpts:
                for ann_k in img_k:
                    image_id = ann_k['image_id']
                    sorted_ids.append(image_id)
                    pred_probs.append(ann_k['prob'])
                    pred_kpts = ann_k['keypoints']
                    if isinstance(pred_kpts, np.ndarray):
                        pred_kpts = pred_kpts.flatten().tolist()
                    pred_confs.append(pred_kpts[2::3])
                    ann = self.coco.imgToAnns[image_id][0]
                    gt_probs.append(ann['keypoints'][2::3])
            gt_probs = np.array(gt_probs).flatten().astype(float)
            pred_probs = np.array(pred_probs).flatten()
            pred_confs = np.array(pred_confs).flatten()
            
            info_str.extend(self._eval_keypoints_classification(
                gt_probs, pred_probs, prefix="io_prob_", balanced=True, verbose=True,
                zeros=[3], ones=[1, 2], nan=[0],
            ))
            info_str.extend(self._eval_keypoints_classification(
                gt_probs, pred_probs, prefix="vo_prob_", balanced=True, verbose=True,
                zeros=[3], ones=[2], nan=[0, 1],
            ))
            info_str.extend(self._eval_keypoints_classification(
                gt_probs, pred_confs, prefix="io_conf_", balanced=True, verbose=True,
                zeros=[3], ones=[1, 2], nan=[0],
            ))
            info_str.extend(self._eval_keypoints_classification(
                gt_probs, pred_confs, prefix="vo_conf_", balanced=True, verbose=True,
                zeros=[3], ones=[2], nan=[1, 0],
            ))

            if "io_prob_best_thr" in dict(info_str):
                best_prob_thr = dict(info_str)["io_prob_best_thr"]
            if "io_conf_best_thr" in dict(info_str):
                best_conf_thr = dict(info_str)["io_conf_best_thr"]
            
        
        eval_params = [
            {"prefix": "", "match_by_bbox": False, "extended": False, "threshold": best_conf_thr},
        ]

        if len(self.coco.imgs) == len(self.coco.anns):
            eval_params.extend([
                {"prefix": "ConfEx_", "match_by_bbox": False, "extended": True, "threshold": best_conf_thr},
                {"prefix": "ProbEx_", "match_by_bbox": False, "extended": True, "threshold": best_prob_thr},
            ])
        else:
            eval_params.append(
                {"prefix": "NoMtch_", "match_by_bbox": True, "extended": False, "threshold": best_conf_thr},
            )
        
        for param in eval_params:
            print("\n", "+"*40)
            print(f"Eval params: {param}")
            det_copy = deepcopy(coco_det)
            
            if "prob" in param["prefix"].lower():
                # Replace the confidence by the probability
                for ann in det_copy.dataset["annotations"]:
                    probs = None
                    for img_k in kpts:
                        for ann_k in img_k:
                            if ann_k["image_id"] == ann["image_id"]:
                                probs = ann_k["prob"]
                                break
                    if isinstance(probs, np.ndarray):
                        probs = probs.flatten().tolist()
                    if not probs is None:
                        ann["keypoints"][2::3] = probs

            self.coco_eval = COCOeval(
                deepcopy(self.coco),
                det_copy,
                'keypoints',
                self.sigmas,
                match_by_bbox = param["match_by_bbox"],
                extended_oks = param["extended"],
                confidence_thr = param["threshold"],
                # alpha = 0.0,
                # beta = (1 - np.exp(-1)) / (2 - np.exp(-1)),
                # beta = 0.0,
                )
            self.coco_eval.params.useSegm = None
            self.coco_eval.evaluate()
            self.coco_eval.accumulate()
            self.coco_eval.summarize()

            if return_wrong_images:
                sorted_matches, sort_idx = self._sort_images_by_prediction_score(self.coco_eval)

            try:
                stats_names = self.coco_eval.stats_names
            except AttributeError:
                stats_names = [
                    'AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5',
                    'AR .75', 'AR (M)', 'AR (L)'
                ]
            stats_names = [param["prefix"] + s for s in stats_names]
            new_info_str = list(zip(stats_names, self.coco_eval.stats))
            
            # If not vanilla eval, filter out .5, .75, M, L and AR values
            if param["prefix"] != "":
                new_info_str = [
                    x for x in new_info_str if (
                        '.' not in x[0] and
                        'M' not in x[0] and
                        'L' not in x[0] and
                        'S' not in x[0] and 
                        'AR' not in x[0])
                ]
            info_str.extend(new_info_str)
            # break

        # Remove all tuples with '.' in the first element
        info_str = [x for x in info_str if '.' not in x[0]]

        if return_wrong_images:
            return info_str, sorted_matches, sort_idx
        else:
            return info_str
        
    def _sort_images_by_prediction_score(self, coco_eval):
        # print("\n\nSort by predition score\n\n")
        if not hasattr(coco_eval, "matched_pairs"):
            coco_eval.evaluate()

        matches = np.array(coco_eval.matched_pairs)

        # breakpoint()

        ious = [m[2] for m in matches]
        sort_idx = np.argsort(ious)
        sorted_matches = matches[sort_idx]

        return sorted_matches, sort_idx

    def _sort_and_unique_bboxes(self, kpts, key='bbox_id'):
        """sort kpts and remove the repeated ones."""
        for img_id, persons in kpts.items():
            num = len(persons)
            kpts[img_id] = sorted(kpts[img_id], key=lambda x: x[key])
            for i in range(num - 1, 0, -1):
                if kpts[img_id][i][key] == kpts[img_id][i - 1][key]:
                    del kpts[img_id][i]

        return kpts
