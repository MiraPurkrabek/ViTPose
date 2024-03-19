# Copyright (c) OpenMMLab. All rights reserved.
import os
from tqdm import tqdm
import numpy as np
import mmcv
from mmcv.runner import load_checkpoint
import warnings
from argparse import ArgumentParser

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)
from mmpose.models import build_posenet
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

from posevis import pose_visualization

def my_init_pose_model(config, checkpoint=None, device='cuda:0'):
    """Initialize a pose model from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    model = build_posenet(config.model)
    if checkpoint is not None:
        # load model checkpoint
        load_checkpoint(model, checkpoint, map_location='cpu')
    # save the config in the model for convenience
    model.cfg = config
    model.to(device)
    model.eval()
    return model


def bbox_from_kpts(kpts, old_bbox, format='xyxy', padding=1.0):
    """Check if the keypoints are inside the bounding box.

    Args:
        kpts (np.ndarray): The shape is N x 3. The first two dimensions are
            the coordinates and the third dimension is the score.
        bbox (np.ndarray): The shape is 4. The four dimensions are x1, y1, x2,
            y2.

    Returns:
        np.ndarray: The modified bounding box that contains all keypoints.
    """
    min_x = np.min(kpts[:, 0])
    max_x = np.max(kpts[:, 0])
    min_y = np.min(kpts[:, 1])
    max_y = np.max(kpts[:, 1])
    
    center = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2])
    width = max_x - min_x
    height = max_y - min_y

    # Pad the bbox
    width *= padding
    height *= padding

    bbox = np.array([
        center[0] - width / 2,
        center[1] - height / 2,
        center[0] + width / 2,
        center[1] + height / 2,
    ])

    # The new bbox cannot be smaller than the old one
    bbox[:2] = np.minimum(bbox[:2], old_bbox[:2])
    bbox[2:4] = np.maximum(bbox[2:4], old_bbox[2:4])
    
    if format.lower() == "xywh":
        bbox = np.array([
            bbox[0],
            bbox[1],
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
        ])
    elif format.lower() == "cs":
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        bbox = np.array([
            bbox[0] + width / 2,
            bbox[1] + height / 2,
            width,
            height,
        ])
    
    return bbox


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--img-root', type=str, default='', help='Image root')
    parser.add_argument('--img', type=str, default='', help='Image file')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--out-img-root',
        type=str,
        default='',
        help='root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
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

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.show or (args.out_img_root != '')
    assert args.img != '' or args.img_root != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
    # build the pose model from a config file and a checkpoint file
    pose_model = my_init_pose_model(
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

    image_name = os.path.join(args.img_root, args.img)
    print("Image name -", image_name)
    if os.path.isdir(image_name):
        print("Running the mmpose on the whole folder")
        images_names = list(map(
            lambda x: os.path.join(image_name, x),
            [dr for dr in os.listdir(image_name) if (
                os.path.isfile(os.path.join(image_name, dr)) and 
                (dr.lower().endswith(".jpg") or dr.lower().endswith(".png") or dr.lower().endswith(".jpeg"))
            )]
        ))
    else:
        print("Running the mmpose on a single image")
        images_names = [image_name]

    results = []

    for img_i, image_name in enumerate(tqdm(images_names, ascii=True)):
        _, relative_image_name = os.path.split(image_name)

        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(det_model, image_name)

        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, args.det_cat_id)

        # Select exactly N persons (the one with the most confidence)
        N = 1
        person_results = person_results[:N]
        for i in range(len(person_results)):
            person_results[i]["bbox"][4] = 1.0
            
        if len(person_results) == 0:
            print("No person detected in the image", image_name)
            continue

        # test a single image, with a list of bboxes.

        # optional
        return_heatmap = False

        # e.g. use ('backbone', ) to return backbone feature
        output_layer_names = None

        stable_bbox = False
        idx = 0
        while not stable_bbox:
            # pose_results = person_results
            pose_results, returned_outputs = inference_top_down_pose_model(
                pose_model,
                image_name,
                person_results,
                bbox_thr=args.bbox_thr,
                format='xyxy',
                dataset=dataset,
                dataset_info=dataset_info,
                return_heatmap=return_heatmap,
                outputs=output_layer_names)
            
            # Check if the bounding box is stable
            old_bbox = person_results[0]["bbox"][:4]
            kpts_bbox = bbox_from_kpts(
                pose_results[0]["keypoints"][:, :3],
                old_bbox,
                format="xyxy",
                padding=1.05)
            stable_bbox = np.abs(kpts_bbox - old_bbox)
            stable_bbox = (stable_bbox < 0.01 * old_bbox).all()
            
            if not stable_bbox:
                person_results[0]["bbox"][:4] = kpts_bbox
            idx +=1

            if idx > 10:
                print("The bounding box is not stable even after 10 iterations for image '{}'. Exiting the loop.".format(
                    image_name
                ))
                break

        if args.out_img_root == '':
            out_file = None
        else:
            os.makedirs(args.out_img_root, exist_ok=True)
            out_file = os.path.join(
                args.out_img_root,
                "vis_{:s}".format(relative_image_name)
            )

        # Show the results using my visualization
        for pose_result in pose_results:
            if 'keypoints' in pose_result:
                kpts = pose_result['keypoints']
                kpts[kpts[:, 2] >= args.kpt_thr, 2] = 2
                kpts[kpts[:, 2] < args.kpt_thr, 2] = 0
                pose_result['keypoints'] = kpts
            pose_result['bbox'] = pose_result['bbox'][:4]
            pose_result['bbox'][2:] = pose_result['bbox'][2:] - pose_result['bbox'][:2]
            pose_result["image_name"] = relative_image_name

            results.append(pose_result)
        try:
            save_img = pose_visualization(
                    image_name,
                    pose_results,
                    show_markers=True,
                    line_type="solid",
                    width_multiplier=3.0,
                    show_bbox=False,
                    differ_individuals=False,
                )
            mmcv.image.imwrite(save_img, out_file)
        except:
            print("Error while saving the image", image_name)

        # # show the results
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
            
        # if img_i == 10:
        #     break

    # Save the results into a json file
    print("Saving the results into a json file")
    mmcv.dump(results, os.path.join(args.out_img_root, "results.json"))


if __name__ == '__main__':
    main()
