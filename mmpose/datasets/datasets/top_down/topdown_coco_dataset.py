# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict, defaultdict

import json_tricks as json
import numpy as np
from mmcv import Config, deprecated_api_warning
# from xtcocotools.cocoeval import COCOeval
from ._cocoeval import COCOeval

from ....core.post_processing import oks_nms, soft_oks_nms
from ...builder import DATASETS
from ..base import Kpt2dSviewRgbImgTopDownDataset

import time
import datetime

########################################################################################
########################################################################################

def accumulate_orig(self, p = None):
    '''
    Accumulate per image evaluation results and store the result in self.eval
    :param p: input params for evaluation
    :return: None
    '''
    print('Accumulating evaluation results...')
    tic = time.time()
    if not self.evalImgs:
        print('Please run evaluate() first')
    # allows input customized parameters
    if p is None:
        p = self.params
    p.catIds = p.catIds if p.useCats == 1 else [-1]
    T           = len(p.iouThrs)
    R           = len(p.recThrs)
    K           = len(p.catIds) if p.useCats else 1
    A           = len(p.areaRng)
    M           = len(p.maxDets)
    precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
    recall      = -np.ones((T,K,A,M))
    scores      = -np.ones((T,R,K,A,M))

    # create dictionary for future indexing
    _pe = self._paramsEval
    catIds = _pe.catIds if _pe.useCats else [-1]
    setK = set(catIds)
    setA = set(map(tuple, _pe.areaRng))
    setM = set(_pe.maxDets)
    setI = set(_pe.imgIds)
    # get inds to evaluate
    k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
    m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
    a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
    i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
    I0 = len(_pe.imgIds)
    A0 = len(_pe.areaRng)
    # retrieve E at each category, area range, and max number of detections
    for k, k0 in enumerate(k_list):
        Nk = k0*A0*I0
        for a, a0 in enumerate(a_list):
            Na = a0*I0
            for m, maxDet in enumerate(m_list):
                print("\n===")
                E = [self.evalImgs[Nk + Na + i] for i in i_list]
                E = [e for e in E if not e is None]
                if len(E) == 0:
                    continue
                dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])
                # different sorting method generates slightly different results.
                # mergesort is used to be consistent as Matlab implementation.
                inds = np.argsort(-dtScores, kind='mergesort')
                dtScoresSorted = dtScores[inds]

                # print(self.params.imgIds)
                i = 0
                for imgId in self.params.imgIds:
                    oks = self.computeOks(imgId, 1)
                    if len(oks) > 0:
                        print("{} ({:d}): {}".format(imgId, i, oks))
                        i+=1

                # for e in E:
                #     print("{}: {} x {}".format(e["image_id"], e["dtIds"], e["gtIds"]))
                # print("E[0]", E[0])

                dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
                print("dtm.shape", dtm.shape)
                dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                gtIg = np.concatenate([e['gtIgnore'] for e in E])
                npig = np.count_nonzero(gtIg == 0)
                if npig == 0:
                    continue
                # https://github.com/cocodataset/cocoapi/pull/332/
                tps = np.logical_and(dtm >= 0, np.logical_not(dtIg))
                fps = np.logical_and(dtm < 0, np.logical_not(dtIg))

                print("tps.shape", tps.shape)
                print("fps.shape", fps.shape)
                
                print("fps[0] == fps[3]", np.all(fps[0, :] == fps[3, :]))
                print("tps[3]", tps[3, :])
                print("fps[3]", fps[3, :])

                ious_keys = {k:v for k, v in self.ious.items() if v != []}
                print("ious_keys", len(self.ious.keys()))
                for k, v in ious_keys.items():
                    print("{}: {}".format(k, v.shape))

                
                
                tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float64)
                fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float64)
                
                print("tp_sum.shape", tp_sum.shape)
                print("fp_sum.shape", fp_sum.shape)
                
                for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                    tp = np.array(tp)
                    fp = np.array(fp)
                    nd = len(tp)
                    rc = tp / npig
                    pr = tp / (fp+tp+np.spacing(1))
                    q  = np.zeros((R,))
                    ss = np.zeros((R,))

                    if nd:
                        recall[t,k,a,m] = rc[-1]
                    else:
                        recall[t,k,a,m] = 0

                    # numpy is slow without cython optimization for accessing elements
                    # use python array gets significant speed improvement
                    pr = pr.tolist(); q = q.tolist()

                    for i in range(nd-1, 0, -1):
                        if pr[i] > pr[i-1]:
                            pr[i-1] = pr[i]

                    inds = np.searchsorted(rc, p.recThrs, side='left')
                    try:
                        for ri, pi in enumerate(inds):
                            q[ri] = pr[pi]
                            ss[ri] = dtScoresSorted[pi]
                    except:
                        pass
                    precision[t,:,k,a,m] = np.array(q)
                    scores[t,:,k,a,m] = np.array(ss)
    self.eval = {
        'params': p,
        'counts': [T, R, K, A, M],
        'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'precision': precision,
        'recall':   recall,
        'scores': scores,
    }
    toc = time.time()
    print('DONE (t={:0.2f}s).'.format( toc-tic))

########################################################################################
########################################################################################


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
        img_ann = self.coco.loadImgs(img_id)[0]
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
            x2 = min(width - 1, x1 + max(0, w))
            y2 = min(height - 1, y1 + max(0, h))
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
            joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3])

            image_file = osp.join(self.img_prefix, self.id2name[img_id])
            rec.append({
                'image_file': image_file,
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

        if isinstance(all_boxes, dict):
            all_boxes = all_boxes["annotations"]

        print(f'=> Total boxes: {len(all_boxes)}')

        kpt_db = []
        bbox_id = 0
        for det_res in all_boxes:
            if det_res['category_id'] != 1:
                continue

            image_file = osp.join(self.img_prefix,
                                  self.id2name[det_res['image_id']])
            box = det_res['bbox']

            try:
                score = det_res['score']
            except KeyError:
                score = 1.0

            if score < self.det_bbox_thr:
                continue

            joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d_visible = np.ones((num_joints, 3), dtype=np.float32)
            kpt_db.append({
                'image_file': image_file,
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
                - image_paths (list[str]): For example, ['/datagrid/personal/purkrmir/data/COCO/original/val2017\
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

        for result in results:
            preds = result['preds']
            boxes = result['boxes']
            image_paths = result['image_paths']
            bbox_ids = result['bbox_ids']

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
                    'bbox_id': bbox_ids[i]
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
                if kwargs.get('rle_score', False):
                    pose_score = n_p['keypoints'][:, 2]
                    n_p['score'] = float(box_score + np.mean(pose_score) +
                                         np.max(pose_score))
                else:
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

        self._write_coco_keypoint_results(valid_kpts, res_file)

        # do evaluation only if the ground truth keypoint annotations exist
        if 'annotations' in self.coco.dataset:
            info_str, wrong_ids, indices, sample_score = self._do_python_keypoint_eval(res_file, return_wrong_images=True)
            name_value = OrderedDict(info_str)

            wrong_paths = np.array(list(map(
                lambda f: osp.join(self.img_prefix, "{:012d}.jpg".format(f)),
                wrong_ids,
            )))

            if tmp_folder is not None:
                tmp_folder.cleanup()
        else:
            warnings.warn(f'Due to the absence of ground truth keypoint'
                          f'annotations, the quantitative evaluation can not'
                          f'be conducted. The prediction results have been'
                          f'saved at: {osp.abspath(res_file)}')
            name_value = {}

        if return_score:
            return name_value, wrong_paths, indices, sample_score
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

    def _do_python_keypoint_eval(self, res_file, return_wrong_images=False):
        """Keypoint evaluation using COCOAPI."""
        coco_det = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_det, 'keypoints', self.sigmas)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        # print(coco_eval.ious)
        # evalImgs = np.array(coco_eval.evalImgs)
        # evalImgs = evalImgs[evalImgs != None]
        # print(len(evalImgs))
        # coco_eval = accumulate(coco_eval)
        # accumulate_orig(coco_eval)
        coco_eval.accumulate()
        # print(coco_eval.eval["precision"].shape, coco_eval.eval["recall"].shape)
        coco_eval.summarize()

        if return_wrong_images:
            wrong_image_ids, img_score, sample_score = self._sort_images_by_prediction_score(coco_eval)

        stats_names = [
            'AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5',
            'AR .75', 'AR (M)', 'AR (L)'
        ]

        info_str = list(zip(stats_names, coco_eval.stats))

        if return_wrong_images:
            return info_str, wrong_image_ids, img_score, sample_score
        else:
            return info_str

    def _sort_images_by_prediction_score(self, coco_eval):
        if not hasattr(coco_eval, "ious"):
            coco_eval.evaluate()

        sample_score = np.array([])
        img_score = np.ones(len(coco_eval.params.imgIds), dtype=np.float) * np.nan

        # Take only eval images with area == all --> first part of the evalImgs
        n_areas = len(coco_eval.params.areaRng)
        eval_imgs_slice = len(coco_eval.evalImgs) // n_areas
        eval_imgs = coco_eval.evalImgs[:eval_imgs_slice]

        # Compute score for each image. Score is MIN / MEAN of IoUs (= OKSs) 
        # over all poses in the image
        num_nans = 0
        num_samples = 0
        for score_i, key in enumerate(coco_eval.ious.keys()):
            img_ious = np.array(coco_eval.ious[key])
            img_eval = eval_imgs[score_i]
            
            if len(img_ious) == 0:
                continue

            # print("-"*20)
            # print(img_ious, img_ious.shape)
            # print(img_eval)
            gt_ignore = img_eval["gtIgnore"].astype(bool).flatten()
            # print(gt_ignore, gt_ignore.shape)
            img_ious = img_ious[:, ~gt_ignore]

            # Add np.nan for all GTs that are ignored
            sample_score = np.append(sample_score, np.ones(np.sum(gt_ignore)) * np.nan)

            if len(img_ious) == 0:
                # sample_score = np.append(sample_score, np.ones(np.sum(gt_ignore)) * np.nan)
                print("All GTs ignored", img_eval["image_id"])
                continue

            best_ious = np.max(img_ious, axis=0)
            num_samples += len(best_ious)
            idx_0 = np.argmax(img_ious, axis=0)
            idx_0_unique = np.unique(idx_0)

            # if len(best_ious) > 1:
            #     print("#"*30)
            #     print(img_eval)
            #     print(img_ious)
            #     print(best_ious)
            #     print(idx_0)
            #     print(idx_0_unique)

            img_score[score_i] = np.mean(best_ious)
            sample_score = np.append(sample_score, best_ious)

            # assert len(idx_0) == len(idx_0_unique)

            # None == no annotations for this image
            # if img_eval is None:
            #     score[score_i] = np.nan
            #     num_nans += 1
            #     continue
            
            # dt_ids = np.array(img_eval['dtIds'])
            # gt_ids = np.array(img_eval['gtIds'])
            # gt_matches = np.array(img_eval['gtMatches'])
            # dt_matches = np.array(img_eval['dtMatches'])

            # # print("-----")
            # # print(gt_ids, dt_ids)
            # # print(gt_matches, dt_matches)

            # num_preds = len(dt_ids)
            # num_gt = len(gt_ids)
            
            # # No GT --> 'empty' image, high score
            # # (as we are interested in the worst images and these are technically right)
            # if num_preds == 0 and num_gt == 0:
            #     score[score_i] = np.nan
            #     continue
            
            # # Either GT or PRED amount is 0 --> score 0 as there are only FP or FN
            # elif num_gt == 0 or num_preds == 0:
            #     score[score_i] = 0
            #     continue
            
            # # Filter by ignore masks
            # gt_ignore_mask = ~ img_eval["gtIgnore"].astype(bool)
            # gt_ids = gt_ids[gt_ignore_mask]
            # gt_matches = gt_matches[0, gt_ignore_mask]
            # dt_ignore_mask = ~ img_eval["dtIgnore"][0].astype(bool)
            # dt_ids = dt_ids[dt_ignore_mask]
            # dt_matches = dt_matches[0, dt_ignore_mask]
            
            # # print("-----")
            # # print(gt_ids, dt_ids)
            # # print(gt_matches, dt_matches)

            # img_ious = img_ious[dt_ignore_mask, :]
            # img_ious = img_ious[:, gt_ignore_mask]

            # # Filter matches that are '-1'
            # gt_nonmatch_mask = gt_matches >= 0
            # dt_nonmatch_mask = dt_matches >= 0
            # gt_ids = gt_ids[gt_nonmatch_mask]
            # dt_ids = dt_ids[dt_nonmatch_mask]
            # gt_matches = gt_matches[gt_nonmatch_mask]
            # dt_matches = dt_matches[dt_nonmatch_mask]
            
            # img_ious = img_ious[dt_nonmatch_mask, :]
            # img_ious = img_ious[:, gt_nonmatch_mask]

            # num_FPFN = np.max([np.sum(~dt_nonmatch_mask), np.sum(~gt_nonmatch_mask)]).squeeze()
            
            # num_preds = len(dt_ids)
            # num_gt = len(gt_ids)

            # # No GT --> 'empty' image, high score
            # # (as we are interested in the worst images and these are technically right)
            # if num_gt == 0 and num_preds == 0 and num_FPFN > 0:
            #     # Somehow, GT and DT were not matched - OKS was so low
            #     tentative_score = -1

            # # Exactly one GT and one PRED for the image
            # # Take their IoU (= OKS) as score
            # elif num_gt == 1 and num_preds == 1:
            #     tentative_score = [img_ious.squeeze()]

            # # There is the same number of GT and PRED. Parse them and compute 
            # # MIN / MEAN of their IoU (= OKS)
            # elif num_gt == num_preds:
            #     row_idx1 = np.argsort(dt_matches)
            #     col_idx1 = list(range(num_gt))
            #     tentative_score = img_ious[row_idx1, col_idx1].squeeze()
            #     # row_idx2 = list(range(num_preds))
            #     # col_idx2 = np.argsort(gt_matches)
            #     # score2 = np.mean(img_ious[row_idx2, col_idx2]).squeeze()
            #     # print("="*20)
            #     # print("Same number, need parsing")
            #     # print(num_gt, num_preds)
            #     # print(gt_ids, dt_ids)
            #     # print(gt_matches)
            #     # print(dt_matches)

            #     # # print(row_idx, col_idx)
            #     # print("+"*10)
            #     # print(img_ious)
            #     # print("+"*10)
            #     # print(img_ious[row_idx1, col_idx1], score1)
            #     # print(img_ious[row_idx2, col_idx2], score2)

            # # At least some of the persons were mis-predicted.
            # # Compute MEAN / MIN over all poses where mis-predicted pose is 0
            # else: 
            #     raise ValueError("Different numbers")

            # # tentative_score = np.concatenate([tentative_score, np.zeros(1, num_FPFN)])
            # tentative_score = np.array(tentative_score).flatten()
            # if len(tentative_score) == 0:
            #     score[score_i] = 0
            # else:
            #     score[score_i] = np.mean(tentative_score).squeeze()
                            

        # score = np.zeros(len(coco_eval.params.imgIds), dtype=np.float)
        # E = [e for e in coco_eval.evalImgs if not e is None]
        # for ii, imgId in enumerate(coco_eval.params.imgIds):
        #     oks = coco_eval.computeOks(imgId, 1)
        #     if len(oks) > 0:
        #         gt_matches = E[i]["gtMatches"][0, :]
        #         dtIds = E[i]["dtIds"]

        #         try:
        #             new_oks = []
        #             for i in range(len(dtIds)):
        #                 if gt_matches[i] >= 0:
        #                     new_oks.append(oks[i, np.where(gt_matches[i] == dtIds)])
        #                 else:
        #                     new_oks.append(100)
        #             new_oks = np.array(new_oks)
        #         except Exception as e:
        #             new_oks = np.array([10])
        #             # print(e)
        #             # print("===")
        #             # print(gt_matches)
        #             # print(dtIds)
        #             # print(oks)
        #         #     score[ii] = 10

        #         if new_oks.size == 0:
        #             # Images where the detection is completely wrong (below threshold)
        #             new_oks = np.array([0])

        #         new_oks = new_oks.squeeze()
        #         score[ii] = np.mean(new_oks)
                
        #         i+=1
        #     else:
                # Images not 'mentioned' in the annotation
                # score[ii] = 10

        ind = np.argsort(img_score)
        img_score = img_score[ind]
        sorted_images = np.array(coco_eval.params.imgIds)[ind]

        # print(score[:50])

        return sorted_images, img_score, sample_score

    def _sort_and_unique_bboxes(self, kpts, key='bbox_id'):
        """sort kpts and remove the repeated ones."""
        for img_id, persons in kpts.items():
            num = len(persons)
            kpts[img_id] = sorted(kpts[img_id], key=lambda x: x[key])
            for i in range(num - 1, 0, -1):
                if kpts[img_id][i][key] == kpts[img_id][i - 1][key]:
                    del kpts[img_id][i]

        return kpts
