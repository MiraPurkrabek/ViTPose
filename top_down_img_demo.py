# Copyright (c) OpenMMLab. All rights reserved.
import os
import json
import warnings
from argparse import ArgumentParser

from xtcocotools.coco import COCO

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo
import mmcv
from posevis import pose_visualization
from tqdm import tqdm
from copy import deepcopy

import cv2
import numpy as np

def main():
    """Visualize the demo images.

    Require the json_file containing boxes.
    """
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for detection')
    parser.add_argument('pose_checkpoint', help='Checkpoint file')
    parser.add_argument('--img-root', type=str, default='', help='Image root')
    parser.add_argument(
        '--json-file',
        type=str,
        default='',
        help='Json file containing image info.')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--out-img-root',
        type=str,
        default='',
        help='Root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    args = parser.parse_args()

    assert args.show or (args.out_img_root != '')

    coco = COCO(args.json_file)
    coco_data = json.load(open(args.json_file, 'r'))
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    img_keys = list(coco.imgs.keys())

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    coco_data["annotations"] = []

    # process each image
    for i in tqdm(range(len(img_keys)), ascii=True):
        # get bounding box annotations
        image_id = img_keys[i]
        image = coco.loadImgs([image_id])[0]
        image_name = os.path.join(args.img_root, image['file_name'])
        ann_ids = coco.getAnnIds(image_id)
        _, relative_image_name = os.path.split(image_name)

        # make person bounding boxes
        person_results = []
        for ann_id in ann_ids:
            person = {}
            ann = coco.anns[ann_id]
            # bbox format is 'xywh'
            person['bbox'] = ann['bbox']
            person_results.append(person)

        # test a single image, with a list of bboxes
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            image_name,
            person_results,
            bbox_thr=None,
            format='xywh',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        if args.out_img_root == '':
            out_file = None
        else:
            os.makedirs(args.out_img_root, exist_ok=True)
            out_file = os.path.join(
                args.out_img_root,
                "vis_{:s}".format(relative_image_name)
            )
            out_file_blur = os.path.join(
                args.out_img_root,
                "visblur_{:s}".format(relative_image_name)
            )

        # Show the results using my visualization
        for pose_result, ann_id in zip(pose_results, ann_ids):
            kpts = pose_result['keypoints']
            kpts[kpts[:, 2] >= args.kpt_thr, 2] = 2
            kpts[kpts[:, 2] < args.kpt_thr, 2] = 0
            pose_result['keypoints'] = kpts

            save_ann = deepcopy(pose_result)
            save_ann["keypoints"] = save_ann["keypoints"].flatten().tolist()
            save_ann["bbox"] = save_ann["bbox"].flatten().tolist()
            save_ann["bbox"][2] -= save_ann["bbox"][0]
            save_ann["bbox"][3] -= save_ann["bbox"][1]
            save_ann["image_id"] = image_id
            save_ann["category_id"] = 1
            save_ann["id"] = ann_id
            coco_data["annotations"].append(save_ann)

        try:
            save_img = pose_visualization(
                    image_name,
                    pose_results,
                    show_markers=True,
                    line_type="solid",
                    width_multiplier=1.0,
                    show_bbox=True,
                )
            img = mmcv.imread(image_name)
            kernel_size = np.max(img.shape[:2]) // 5
            if kernel_size % 2 == 0:
                kernel_size += 1
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            save_img_blur = pose_visualization(
                    img,
                    pose_results,
                    show_markers=True,
                    line_type="solid",
                    width_multiplier=1.0,
                    show_bbox=True,
                )
            mmcv.image.imwrite(save_img, out_file)
            mmcv.image.imwrite(save_img_blur, out_file_blur)
        except Exception as e:
            print("Error while saving the image", image_name)
            print(e)

        # vis_pose_result(
        #     pose_model,
        #     image_name,
        #     pose_results,
        #     dataset=dataset,
        #     dataset_info=dataset_info,
        #     kpt_score_thr=args.kpt_thr,
        #     radius=args.radius,
        #     thickness=args.thickness,
        #     show=args.show,
        #     out_file=out_file)

    # Save the coco_data
    new_json_file = args.json_file.replace(".json", "_with_pose.json")
    with open(new_json_file, 'w') as f:
        json.dump(coco_data, f, indent=2)


if __name__ == '__main__':
    main()
