import argparse
import json
import os
import numpy as np
import cv2

def parse_arguments():
    parser = argparse.ArgumentParser(description='Read COCO annotation file')
    parser.add_argument(
        '--annotation_file',
        type=str,
        # default="/datagrid/personal/purkrmir/data/CrowdedPose/COCO-like/annotations/person_keypoints_val2017.json",
        default="/datagrid/personal/purkrmir/data/COCO/original/annotations/person_keypoints_val2017.json",
        help='Path to COCO annotation file',
    )
    parser.add_argument(
        '--path_to_images',
        type=str,
        # default="/datagrid/personal/purkrmir/data/CrowdedPose/COCO-like/val2017/",
        default="/datagrid/personal/purkrmir/data/COCO/original/val2017/",
        help='Path to COCO annotation file',
    )
    return parser.parse_args()

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

def in_bbox(kpts, bbox, bbox_format="xywh"):
    """Check whether the keypoints are inside the bbox.

    Args:
        kpts (ndarray): Keypoints in shape (N, 2).
        bbox (ndarray): Bbox in shape (4,). (left, top, width, height)
        bbox_format (str): Bbox format. Default: 'xywh'.

    Returns:
        list[bool]: List of bool denoting whether the keypoint is inside the
            bbox.
    """

    assert bbox_format in ["xywh", "xyxy"]
    assert kpts.shape[1] >= 2
    assert bbox.shape[0] == 4

    if bbox_format == "xywh":
        bbox = bbox.copy()
        bbox[2:] += bbox[:2]

    x_in = np.logical_and(kpts[:, 0] >= bbox[0], kpts[:, 0] <= bbox[2])
    y_in = np.logical_and(kpts[:, 1] >= bbox[1], kpts[:, 1] <= bbox[3])

    return np.logical_and(x_in, y_in)

def read_annotation_file(annotation_file):
    with open(annotation_file, 'r') as file:
        content = json.load(file)
    return content

def visualize_annotation(annotation, image_path):
    random_crop = np.random.rand() * 0.5 + 1.2

    
    # print(annotation)
    print(image_path)
    image = cv2.imread(image_path)

    kpts = np.array(annotation['keypoints']).reshape(-1, 3)
    vis = kpts[:, 2]
    valid_kpts = vis > 0
    bbox = np.array(annotation["bbox"]).astype(int)
    bbox_w_aspect = np.array([
        bbox[0], bbox[1],
        bbox[2]/1.3, bbox[3]
    ])
    center, scale = bbox_xywh2cs(bbox, 3/4, 1.0)
    shift_x = True if scale[1] == bbox[3] else False

    
    bbox_w_aspect = np.array([
        center[0] - scale[0]/2, center[1] - scale[1]/2,
        center[0] + scale[0]/2, center[1] + scale[1]/2
    ]).astype(int)

    ex_scale = scale * 1.25
    extended_bbox = np.array([
        center[0] - ex_scale[0]/2, center[1] - ex_scale[1]/2,
        center[0] + ex_scale[0]/2, center[1] + ex_scale[1]/2
    ]).astype(int)
    
    new_scale = scale / random_crop

    # print(shift_x)
    rnd = np.random.rand()
    # rnd = 0.1
    if rnd < 0.5:
        if shift_x:
            new_bbox = np.array([
                center[0]-bbox[2]/2, bbox_w_aspect[1]+bbox[3]-new_scale[1],
                bbox[2], new_scale[1],
            ]).astype(int)
        else:
            new_bbox = np.array([
                bbox_w_aspect[0]+bbox[2]-new_scale[0], center[1]-bbox[3]/2,
                new_scale[0], bbox[3]
            ]).astype(int)
    else:
        if shift_x:
            new_bbox = np.array([
                center[0]-bbox[2]/2, bbox_w_aspect[1],
                bbox[2], new_scale[1],
            ]).astype(int)
        else:
            new_bbox = np.array([
                bbox_w_aspect[0], center[1]-bbox[3]/2,
                new_scale[0], bbox[3]
            ]).astype(int)

    new_bbox_43_c, new_bbox_43_s = bbox_xywh2cs(new_bbox, 3/4, 1.0)
    new_bbox_43 = np.array([
        new_bbox_43_c[0] - new_bbox_43_s[0]/2, new_bbox_43_c[1] - new_bbox_43_s[1]/2,
        new_bbox_43_c[0] + new_bbox_43_s[0]/2, new_bbox_43_c[1] + new_bbox_43_s[1]/2
    ]).astype(int)

    new_ex_center, new_ex_scale = bbox_xywh2cs(new_bbox, 3/4, 1.25)
    new_extended_bbox = np.array([
        new_ex_center[0] - new_ex_scale[0]/2, new_ex_center[1] - new_ex_scale[1]/2,
        new_ex_center[0] + new_ex_scale[0]/2, new_ex_center[1] + new_ex_scale[1]/2
    ]).astype(int)



    new_bbox[2:] += new_bbox[:2]
    bbox[2:] += bbox[:2]

    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 1)
    cv2.rectangle(image, (bbox_w_aspect[0], bbox_w_aspect[1]), (bbox_w_aspect[2], bbox_w_aspect[3]), (255, 255, 0), 1)
    cv2.rectangle(image, (extended_bbox[0], extended_bbox[1]), (extended_bbox[2], extended_bbox[3]), (0, 0, 255), 1)
    cv2.rectangle(image, (new_bbox[0], new_bbox[1]), (new_bbox[2], new_bbox[3]), (0, 255, 0), 1)
    cv2.rectangle(image, (new_extended_bbox[0], new_extended_bbox[1]), (new_extended_bbox[2], new_extended_bbox[3]), (0, 255, 255), 1)
    cv2.rectangle(image, (new_bbox_43[0], new_bbox_43[1]), (new_bbox_43[2], new_bbox_43[3]), (255, 0, 255), 1)

    in_original = in_bbox(kpts, extended_bbox, bbox_format="xyxy")
    in_new = in_bbox(kpts, new_extended_bbox, bbox_format="xyxy")
    in_tight = in_bbox(kpts, new_bbox, bbox_format="xyxy")


    # z = np.zeros((kpts.shape[0]))
    # dists = np.array([
    #     np.maximum(kpts[:, 0] - bbox[0], z),
    #     np.maximum(kpts[:, 1] - bbox[1], z),
    #     np.maximum(bbox[2] - kpts[:, 0], z),
    #     np.maximum(bbox[3] - kpts[:, 1], z),
    # ])
    # print(dists)
    # dists = np.min(dists, axis=0)
    # dists[~valid_kpts] = 10000
    # print(dists)
    # far_kpt = np.argmin(dists)

    i = -1
    for keypoint, v, i_o, i_n, i_t in zip(kpts, vis, in_original, in_new, in_tight):
        i += 1
        print(i_o, i_n)
        # if i == far_kpt:
        #     cv2.circle(image, (int(keypoint[0]), int(keypoint[1])), 3, (255, 0, 0), -1)
        
        if v == 0:
            continue
        elif not i_t and i_n:
            cv2.circle(image, (int(keypoint[0]), int(keypoint[1])), 3, (0, 255, 255), -1)
        elif not i_n:
            cv2.circle(image, (int(keypoint[0]), int(keypoint[1])), 3, (0, 0, 255), -1)
        else:
            cv2.circle(image, (int(keypoint[0]), int(keypoint[1])), 3, (0, 255, 0), -1)
        
        # cv2.putText(image, str(d), (int(keypoint[0]), int(keypoint[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
    cv2.imwrite("test.jpg", image)


def main():
    args = parse_arguments()
    annotation_content = read_annotation_file(args.annotation_file)
    annotations = annotation_content['annotations']
    images = annotation_content['images']
    imgId2path = {}
    for image in images:
        imgId2path[image['id']] = os.path.join(args.path_to_images, image['file_name'])
    
    rand_idx = np.random.randint(0, len(annotations))
    while not 'keypoints' in annotations[rand_idx] or annotations[rand_idx]['num_keypoints'] < 10:
        rand_idx = np.random.randint(0, len(annotations))
    # rand_idx = 5067
    print(rand_idx)
    visualize_annotation(annotations[rand_idx], imgId2path[annotations[rand_idx]['image_id']])
    
if __name__ == '__main__':
    main()
