# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np

from mmpose.core.post_processing import (affine_transform, fliplr_joints,
                                         get_affine_transform, get_warp_matrix,
                                         warp_affine_joints)
from mmpose.datasets.builder import PIPELINES


@PIPELINES.register_module()
class TopDownRandomFlip:
    """Data augmentation with random image flip.

    Required keys: 'img', 'joints_3d', 'joints_3d_visible', 'center' and
    'ann_info'.

    Modifies key: 'img', 'joints_3d', 'joints_3d_visible', 'center' and
    'flipped'.

    Args:
        flip (bool): Option to perform random flip.
        flip_prob (float): Probability of flip.
    """

    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, results):
        """Perform data augmentation with random image flip."""
        img = results['img']
        joints_3d = results['joints_3d']
        joints_3d_visible = results['joints_3d_visible']
        center = results['center']

        # A flag indicating whether the image is flipped,
        # which can be used by child class.
        flipped = False
        if np.random.rand() <= self.flip_prob:
            flipped = True
            if not isinstance(img, list):
                img = img[:, ::-1, :]
            else:
                img = [i[:, ::-1, :] for i in img]
            if not isinstance(img, list):
                joints_3d, joints_3d_visible = fliplr_joints(
                    joints_3d, joints_3d_visible, img.shape[1],
                    results['ann_info']['flip_pairs'])
                center[0] = img.shape[1] - center[0] - 1
            else:
                joints_3d, joints_3d_visible = fliplr_joints(
                    joints_3d, joints_3d_visible, img[0].shape[1],
                    results['ann_info']['flip_pairs'])
                center[0] = img[0].shape[1] - center[0] - 1

        results['img'] = img
        results['joints_3d'] = joints_3d
        results['joints_3d_visible'] = joints_3d_visible
        results['center'] = center
        results['flipped'] = flipped

        return results


@PIPELINES.register_module()
class TopDownHalfBodyTransform:
    """Data augmentation with half-body transform. Keep only the upper body or
    the lower body at random.

    Required keys: 'joints_3d', 'joints_3d_visible', and 'ann_info'.

    Modifies key: 'scale' and 'center'.

    Args:
        num_joints_half_body (int): Threshold of performing
            half-body transform. If the body has fewer number
            of joints (< num_joints_half_body), ignore this step.
        prob_half_body (float): Probability of half-body transform.
    """

    def __init__(self, num_joints_half_body=8, prob_half_body=0.3):
        self.num_joints_half_body = num_joints_half_body
        self.prob_half_body = prob_half_body

    @staticmethod
    def half_body_transform(cfg, joints_3d, joints_3d_visible):
        """Get center&scale for half-body transform."""
        upper_joints = []
        lower_joints = []
        for joint_id in range(cfg['num_joints']):
            if joints_3d_visible[joint_id][0] > 0:
                if joint_id in cfg['upper_body_ids']:
                    upper_joints.append(joints_3d[joint_id])
                else:
                    lower_joints.append(joints_3d[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        elif len(lower_joints) > 2:
            selected_joints = lower_joints
        else:
            selected_joints = upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)

        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        aspect_ratio = cfg['image_size'][0] / cfg['image_size'][1]

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
        scale = scale * 1.5
        return center, scale

    def __call__(self, results):
        """Perform data augmentation with half-body transform."""
        joints_3d = results['joints_3d']
        joints_3d_visible = results['joints_3d_visible']

        if (np.sum(joints_3d_visible[:, 0]) > self.num_joints_half_body
                and np.random.rand() < self.prob_half_body):

            c_half_body, s_half_body = self.half_body_transform(
                results['ann_info'], joints_3d, joints_3d_visible)

            if c_half_body is not None and s_half_body is not None:
                results['center'] = c_half_body
                results['scale'] = s_half_body

        return results


@PIPELINES.register_module()
class TopDownGetRandomScaleRotation:
    """Data augmentation with random scaling & rotating.

    Required key: 'scale'.

    Modifies key: 'scale' and 'rotation'.

    Args:
        rot_factor (int): Rotating to ``[-2*rot_factor, 2*rot_factor]``.
        scale_factor (float): Scaling to ``[1-scale_factor, 1+scale_factor]``.
        rot_prob (float): Probability of random rotation.
    """

    def __init__(self, rot_factor=40, scale_factor=0.5, rot_prob=0.6):
        self.rot_factor = rot_factor
        self.scale_factor = scale_factor
        self.rot_prob = rot_prob

    def __call__(self, results):
        """Perform data augmentation with random scaling & rotating."""
        s = results['scale']

        sf = self.scale_factor
        rf = self.rot_factor

        s_factor = np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
        s = s * s_factor

        r_factor = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)
        r = r_factor if np.random.rand() <= self.rot_prob else 0

        results['scale'] = s
        results['rotation'] = r

        return results


@PIPELINES.register_module()
class RandomBlackMask:
    """Mask random parts of the image with black.

    Required keys:'img', 'joints_3d', 'joints_3d_visible'.

    Modified keys:'img', 'joints_3d_visible'.

    Args:
        min_mask (float): Minimum part of the image to be masked.
            Default: 0.0
        max_mask (float): Maximum part of the image to be masked.
            Default: 0.5
        mask_prob (float): Probability of masking the image.
            Default: 1.0
    """

    def __init__(self, min_mask=0.0, max_mask=0.5, mask_prob=1.0, min_kpts=6):
        self.min_mask = min_mask
        self.max_mask = max_mask
        self.mask_prob = mask_prob
        self.min_kpts = min_kpts

    def __call__(self, results):
        """Perform data augmentation with random masking."""
        img = results['img']
        kpts = np.array(results['joints_3d'])
        kpts_visible = np.array(results['joints_3d_visible'])
        num_vis_kpts = np.sum(kpts_visible[:, 0] > 0)

        if num_vis_kpts >= self.min_kpts and np.random.rand() < self.mask_prob:

            # Generate random rectangle to keep
            h, w, _ = img.shape
            rh, rw = np.random.uniform(1-self.max_mask, 1-self.min_mask, 2)
            dh = int(h * rh)
            dw = int(w * rw)
            x = np.random.randint(0, w - dw)
            y = np.random.randint(0, h - dh)
            
            # Set all pixels outside of the rectangle to black
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y:y+dh, x:x+dw] = 1
            img[mask == 0] = 0
            results['img'] = img

            # Change the visibility of blacked keypoints to v=1
            in_black = (kpts[:, 0] >= x) & (kpts[:, 0] <= x+dw) & \
                          (kpts[:, 1] >= y) & (kpts[:, 1] <= y+dh)
            kpts_visible[in_black] = 1
            results['joints_3d_visible'] = kpts_visible

        return results


@PIPELINES.register_module()
class TopDownAffine:
    """Affine transform the image to make input.

    Required keys:'img', 'joints_3d', 'joints_3d_visible', 'ann_info','scale',
    'rotation' and 'center'.

    Modified keys:'img', 'joints_3d', and 'joints_3d_visible'.

    Args:
        use_udp (bool): To use unbiased data processing.
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
    """

    def __init__(self, use_udp=False):
        self.use_udp = use_udp

    def __call__(self, results):
        image_size = results['ann_info']['image_size']

        img = results['img']
        joints_3d = results['joints_3d']
        joints_3d_visible = results['joints_3d_visible']
        c = results['center']
        s = results['scale']
        r = results['rotation']
        results["orig_joints_3d"] = joints_3d.copy()

        if self.use_udp:
            trans = get_warp_matrix(r, c * 2.0, image_size - 1.0, s * 200.0)
            if not isinstance(img, list):
                img = cv2.warpAffine(
                    img,
                    trans, (int(image_size[0]), int(image_size[1])),
                    flags=cv2.INTER_LINEAR)
            else:
                img = [
                    cv2.warpAffine(
                        i,
                        trans, (int(image_size[0]), int(image_size[1])),
                        flags=cv2.INTER_LINEAR) for i in img
                ]

            joints_3d[:, 0:2] = \
                warp_affine_joints(joints_3d[:, 0:2].copy(), trans)

        else:
            trans = get_affine_transform(c, s, r, image_size)
            if not isinstance(img, list):
                img = cv2.warpAffine(
                    img,
                    trans, (int(image_size[0]), int(image_size[1])),
                    flags=cv2.INTER_LINEAR)
            else:
                img = [
                    cv2.warpAffine(
                        i,
                        trans, (int(image_size[0]), int(image_size[1])),
                        flags=cv2.INTER_LINEAR) for i in img
                ]
            for i in range(results['ann_info']['num_joints']):
                if joints_3d_visible[i, 0] > 0.0:
                    joints_3d[i,
                              0:2] = affine_transform(joints_3d[i, 0:2], trans)

        results['img'] = img
        results['joints_3d'] = joints_3d
        results['joints_3d_visible'] = joints_3d_visible

        return results


@PIPELINES.register_module()
class TopDownGenerateTarget:
    """Generate the target heatmap.

    Required keys: 'joints_3d', 'joints_3d_visible', 'ann_info'.

    Modified keys: 'target', and 'target_weight'.

    Args:
        sigma: Sigma of heatmap gaussian for 'MSRA' approach.
        kernel: Kernel of heatmap gaussian for 'Megvii' approach.
        encoding (str): Approach to generate target heatmaps.
            Currently supported approaches: 'MSRA', 'Megvii', 'UDP'.
            Default:'MSRA'
        unbiased_encoding (bool): Option to use unbiased
            encoding methods.
            Paper ref: Zhang et al. Distribution-Aware Coordinate
            Representation for Human Pose Estimation (CVPR 2020).
        keypoint_pose_distance: Keypoint pose distance for UDP.
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
        target_type (str): supported targets: 'GaussianHeatmap',
            'CombinedTarget'. Default:'GaussianHeatmap'
            CombinedTarget: The combination of classification target
            (response map) and regression target (offset map).
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
        valid_visibilities (list[int]): Admissible visibility levels.
    """

    def __init__(self,
                 sigma=2,
                 kernel=(11, 11),
                 valid_radius_factor=0.0546875,
                 target_type='GaussianHeatmap',
                 encoding='MSRA',
                 unbiased_encoding=False,
                 valid_visibilities=[1, 2],
                 ignore_zeros=True,
                 normalize=False,
                 probability_map=False,
                 directional_probabilities=False,
                 with_visibility=False):
        self.save_sigma = sigma
        self.unbiased_encoding = unbiased_encoding
        self.kernel = kernel
        self.valid_radius_factor = valid_radius_factor
        self.target_type = target_type
        self.encoding = encoding
        self.valid_visibilities = valid_visibilities
        self.ignore_zeros = ignore_zeros
        self.normalize = normalize
        self.probability_map = probability_map
        self.directional_probabilities = directional_probabilities
        self.with_visibility = with_visibility

    def _msra_generate_target(self, cfg, joints_3d, joints_3d_visible, sigma):
        """Generate the target heatmap via "MSRA" approach.

        Args:
            cfg (dict): data config
            joints_3d: np.ndarray ([num_joints, 3])
            joints_3d_visible: np.ndarray ([num_joints, 3])
            sigma: Sigma of heatmap gaussian
        Returns:
            tuple: A tuple containing targets.

            - target: Target heatmaps.
            - target_weight: (1: visible, 0: invisible)
        """
        num_joints = cfg['num_joints']
        image_size = cfg['image_size']
        W, H = cfg['heatmap_size']
        joint_weights = cfg['joint_weights']
        use_different_joint_weights = cfg['use_different_joint_weights']

        target_weight = np.zeros((num_joints, 1), dtype=np.float32)
        target = np.zeros((num_joints, H, W), dtype=np.float32)

        # 3-sigma rule
        tmp_size = sigma * 3

        if self.unbiased_encoding:
            for joint_id in range(num_joints):
                target_weight[joint_id] = joints_3d_visible[joint_id, 0]

                feat_stride = image_size / [W, H]
                mu_x = joints_3d[joint_id][0] / feat_stride[0]
                mu_y = joints_3d[joint_id][1] / feat_stride[1]
                # Check that any part of the gaussian is in-bounds
                ul = [mu_x - tmp_size, mu_y - tmp_size]
                br = [mu_x + tmp_size + 1, mu_y + tmp_size + 1]
                if ul[0] >= W or ul[1] >= H or br[0] < 0 or br[1] < 0:
                    target_weight[joint_id] = 0

                if target_weight[joint_id] == 0:
                    continue

                x = np.arange(0, W, 1, np.float32)
                y = np.arange(0, H, 1, np.float32)
                y = y[:, None]

                if target_weight[joint_id] > 0.5:
                    target[joint_id] = np.exp(-((x - mu_x)**2 +
                                                (y - mu_y)**2) /
                                              (2 * sigma**2))
        else:
            for joint_id in range(num_joints):
                target_weight[joint_id] = joints_3d_visible[joint_id, 0]

                feat_stride = image_size / [W, H]
                mu_x = int(joints_3d[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints_3d[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= W or ul[1] >= H or br[0] < 0 or br[1] < 0:
                    target_weight[joint_id] = 0

                if target_weight[joint_id] > 0.5:
                    size = 2 * tmp_size + 1
                    x = np.arange(0, size, 1, np.float32)
                    y = x[:, None]
                    x0 = y0 = size // 2
                    # The gaussian is not normalized,
                    # we want the center value to equal 1
                    g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

                    # Usable gaussian range
                    g_x = max(0, -ul[0]), min(br[0], W) - ul[0]
                    g_y = max(0, -ul[1]), min(br[1], H) - ul[1]
                    # Image range
                    img_x = max(0, ul[0]), min(br[0], W)
                    img_y = max(0, ul[1]), min(br[1], H)

                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        if use_different_joint_weights:
            target_weight = np.multiply(target_weight, joint_weights)

        return target, target_weight

    def _megvii_generate_target(self, cfg, joints_3d, joints_3d_visible,
                                kernel):
        """Generate the target heatmap via "Megvii" approach.

        Args:
            cfg (dict): data config
            joints_3d: np.ndarray ([num_joints, 3])
            joints_3d_visible: np.ndarray ([num_joints, 3])
            kernel: Kernel of heatmap gaussian

        Returns:
            tuple: A tuple containing targets.

            - target: Target heatmaps.
            - target_weight: (1: visible, 0: invisible)
        """

        num_joints = cfg['num_joints']
        image_size = cfg['image_size']
        W, H = cfg['heatmap_size']
        heatmaps = np.zeros((num_joints, H, W), dtype='float32')
        target_weight = np.zeros((num_joints, 1), dtype=np.float32)

        for i in range(num_joints):
            target_weight[i] = joints_3d_visible[i, 0]

            if target_weight[i] < 1:
                continue

            target_y = int(joints_3d[i, 1] * H / image_size[1])
            target_x = int(joints_3d[i, 0] * W / image_size[0])

            if (target_x >= W or target_x < 0) \
                    or (target_y >= H or target_y < 0):
                target_weight[i] = 0
                continue

            heatmaps[i, target_y, target_x] = 1
            heatmaps[i] = cv2.GaussianBlur(heatmaps[i], kernel, 0)
            maxi = heatmaps[i, target_y, target_x]

            heatmaps[i] /= maxi / 255

        return heatmaps, target_weight

    def _udp_generate_target(self, cfg, joints_3d, joints_3d_visible, factors,
                             target_type, valid_visibilities=[1, 2], ignore_zeros=True,
                             normalize=False, probability_map=False, directional_probabilities=False):
        """Generate the target heatmap via 'UDP' approach. Paper ref: Huang et
        al. The Devil is in the Details: Delving into Unbiased Data Processing
        for Human Pose Estimation (CVPR 2020).

        Note:
            - num keypoints: K
            - heatmap height: H
            - heatmap width: W
            - num target channels: C
            - C = K if target_type=='GaussianHeatmap'
            - C = 3*K if target_type=='CombinedTarget'

        Args:
            cfg (dict): data config
            joints_3d (np.ndarray[K, 3]): Annotated keypoints.
            joints_3d_visible (np.ndarray[K, 3]): Visibility of keypoints.
            factor (float): kernel factor for GaussianHeatmap target or
                valid radius factor for CombinedTarget.
            target_type (str): 'GaussianHeatmap' or 'CombinedTarget'.
                GaussianHeatmap: Heatmap target with gaussian distribution.
                CombinedTarget: The combination of classification target
                (response map) and regression target (offset map).
            valid_visibilities (list[int]): Admissible visibility levels.

        Returns:
            tuple: A tuple containing targets.

            - target (np.ndarray[C, H, W]): Target heatmaps.
            - target_weight (np.ndarray[K, 1]): (1: visible, 0: invisible)
        """
        num_joints = cfg['num_joints']
        image_size = cfg['image_size']
        heatmap_size = cfg['heatmap_size']
        joint_weights = cfg['joint_weights']
        use_different_joint_weights = cfg['use_different_joint_weights']

        # normalize = normalize or probability_map

        target_weight = np.ones((num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = np.minimum(1, joints_3d_visible[:, 0])
        
        prob_is_ooi = False

        if isinstance(ignore_zeros, float):
            ignore_zeros = np.random.rand() < ignore_zeros
        assert isinstance(ignore_zeros, bool)
        
        if target_type.lower() == 'GaussianHeatmap'.lower():
            if probability_map:
                if directional_probabilities:
                    target = np.zeros((num_joints, heatmap_size[1]* heatmap_size[0] + 4),
                                    dtype=np.float32)
                elif self.with_visibility:
                    target = np.zeros((num_joints, heatmap_size[1]* heatmap_size[0] + 2),
                                    dtype=np.float32)
                    target[:, -2] = int(prob_is_ooi)
                    target[:, -1] = (joints_3d_visible[:, 0].squeeze() == 2).astype(int)
                else:
                    target = np.zeros((num_joints, heatmap_size[1]* heatmap_size[0] + 1),
                                    dtype=np.float32)
                    target[:, -1] = int(prob_is_ooi)
            else:
                target = np.zeros((num_joints, heatmap_size[1], heatmap_size[0]),
                                dtype=np.float32)

            for joint_id in range(num_joints):
                # prepare for gaussian
                if isinstance(factors, list):
                    factor = factors[joint_id]
                else:
                    factor = factors

                if factor < 1e-4:
                    factor = self.save_sigma

                tmp_size = factor * 3
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, None]
                
                target_2d = np.zeros((heatmap_size[1], heatmap_size[0]),
                                     dtype=np.float32)
                vis = int(joints_3d_visible[joint_id, 0])
                # print(joint_id, joints_3d[joint_id, :], joints_3d_visible[joint_id, :])
                # print(vis, vis in valid_visibilities, valid_visibilities)

                # Check that the keypoint visibility is valid
                if vis not in valid_visibilities:
                    # If not, just return the image as is
                    if ignore_zeros:
                        target_weight[joint_id] = 0
                    else:
                        target_weight[joint_id] = min(vis, 0)
                    continue

                feat_stride = (image_size - 1.0) / (heatmap_size - 1.0)
                mu_x = int(joints_3d[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints_3d[joint_id][1] / feat_stride[1] + 0.5)
                
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                     
                # # Generate gaussian
                mu_x_ac = joints_3d[joint_id][0] / feat_stride[0]
                mu_y_ac = joints_3d[joint_id][1] / feat_stride[1]
                x0 = y0 = size // 2
                x0 += mu_x_ac - mu_x
                y0 += mu_y_ac - mu_y
                g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * factor**2))
                
                sum_max_value = 2 * np.pi * factor**2
                max_value = 1
                if normalize:
                    g /= 2 * np.pi * factor**2
                    sum_max_value = 1
                    max_value = 1/(2 * np.pi * factor**2)


                if directional_probabilities:
                    top_prob_ratio = min(abs(heatmap_size[1]/2 - ul[1]) / (2*tmp_size+1), 1)
                    right_prob_ratio = min(abs(heatmap_size[0]/2 - ul[0]) / (2*tmp_size+1), 1)
                    if ul[0] > heatmap_size[0]/2:
                        right_prob_ratio = 0
                    if ul[1] > heatmap_size[1]/2:
                        top_prob_ratio = 0
                    
                    bottom_prob_ratio = 1-top_prob_ratio
                    left_prob_ratio = 1-right_prob_ratio
                    out_of_image_ratios = np.array([
                        top_prob_ratio * right_prob_ratio,
                        top_prob_ratio * left_prob_ratio,
                        bottom_prob_ratio * right_prob_ratio,
                        bottom_prob_ratio * left_prob_ratio
                    ])

                    # breakpoint()
                    assert np.allclose(out_of_image_ratios.sum(), 1)
                    assert np.all(out_of_image_ratios >= 0)
    
                # Check that any part of the gaussian is in-bounds
                if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    
                    # If not, just return the image as is
                    if ignore_zeros:
                        target_weight[joint_id] = 0

                    if directional_probabilities:
                        target[joint_id, -4:] = out_of_image_ratios
                    
                    continue   
                    
                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

                v = vis
                # v = target_weight[joint_id]
                if v > 0.5:
                    target_2d[img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
                
                if probability_map:
                    target_2d = target_2d.flatten()
                    
                    # out_of_image_prob = (target_2d.max() / max_value).round()
                    # out_of_image_prob = (sum_max_value - np.sum(target_2d)) / sum_max_value

                    out_of_image_prob = np.array([1])
                    if mu_x_ac < 0 or mu_x_ac >= heatmap_size[0] or mu_y_ac < 0 or mu_y_ac >= heatmap_size[1]:
                        out_of_image_prob = np.array([0])
                    
                    if prob_is_ooi:
                        out_of_image_prob = 1 - out_of_image_prob

                    if directional_probabilities:
                        out_of_image_prob = out_of_image_ratios * out_of_image_prob
    

                    # tgt_draw = target_2d.reshape(heatmap_size[1], heatmap_size[0]) * 255
                    # tgt_draw -= tgt_draw.min()
                    # tgt_draw /= tgt_draw.max()
                    # tgt_draw *= 255
                    # tgt_draw = tgt_draw.astype(np.uint8)
                    # tgt_draw = cv2.applyColorMap(tgt_draw, cv2.COLORMAP_JET)
                    # cv2.imwrite('TargetTest/target_2d_{}.png'.format(joint_id), tgt_draw)
                    
                    # assert out_of_image_prob >= 0 and out_of_image_prob <= 1
                    target_w_pbt = np.append(target_2d, out_of_image_prob.flatten())

                    if self.with_visibility:
                        v = v == 2
                        target_w_pbt = np.append(target_w_pbt, v)

                    # assert np.allclose(target_w_pbt.sum(), 1)
                    # if not np.allclose(target_w_pbt.sum(), 1):
                    #     print("target_w_pbt.sum() != 1", target_w_pbt.sum())
                    #     print("target_2d.sum()", target_2d.sum())
                    #     print("out_of_image_prob", out_of_image_prob)
                    # assert np.allclose(target_w_pbt.sum(), 1)
                    # assert np.all(target_w_pbt >= 0)
                    target[joint_id] = target_w_pbt
                else:
                    target[joint_id] = target_2d
                    

        elif target_type.lower() == 'CombinedTarget'.lower():
            target = np.zeros(
                (num_joints, 3, heatmap_size[1] * heatmap_size[0]),
                dtype=np.float32)
            feat_width = heatmap_size[0]
            feat_height = heatmap_size[1]
            feat_x_int = np.arange(0, feat_width)
            feat_y_int = np.arange(0, feat_height)
            feat_x_int, feat_y_int = np.meshgrid(feat_x_int, feat_y_int)
            feat_x_int = feat_x_int.flatten()
            feat_y_int = feat_y_int.flatten()
            # Calculate the radius of the positive area in classification
            #   heatmap.
            valid_radius = factors * heatmap_size[1]
            feat_stride = (image_size - 1.0) / (heatmap_size - 1.0)
            for joint_id in range(num_joints):
                mu_x = joints_3d[joint_id][0] / feat_stride[0]
                mu_y = joints_3d[joint_id][1] / feat_stride[1]
                x_offset = (mu_x - feat_x_int) / valid_radius
                y_offset = (mu_y - feat_y_int) / valid_radius
                dis = x_offset**2 + y_offset**2
                keep_pos = np.where(dis <= 1)[0]
                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id, 0, keep_pos] = 1
                    target[joint_id, 1, keep_pos] = x_offset[keep_pos]
                    target[joint_id, 2, keep_pos] = y_offset[keep_pos]
            target = target.reshape(num_joints * 3, heatmap_size[1],
                                    heatmap_size[0])
        else:
            raise ValueError('target_type should be either '
                             "'GaussianHeatmap' or 'CombinedTarget'")

        if use_different_joint_weights:
            target_weight = np.multiply(target_weight, joint_weights)

        return target, target_weight

    def __call__(self, results):
        """Generate the target heatmap."""
        joints_3d = results['joints_3d']
        joints_3d_visible = results['joints_3d_visible']

        self.sigma = results.get('self_sigma', self.save_sigma)

        assert self.encoding in ['MSRA', 'Megvii', 'UDP']

        if self.encoding == 'MSRA':
            if isinstance(self.sigma, list):
                num_sigmas = len(self.sigma)
                cfg = results['ann_info']
                num_joints = cfg['num_joints']
                heatmap_size = cfg['heatmap_size']

                target = np.empty(
                    (0, num_joints, heatmap_size[1], heatmap_size[0]),
                    dtype=np.float32)
                target_weight = np.empty((0, num_joints, 1), dtype=np.float32)
                for i in range(num_sigmas):
                    target_i, target_weight_i = self._msra_generate_target(
                        cfg, joints_3d, joints_3d_visible, self.sigma[i])
                    target = np.concatenate([target, target_i[None]], axis=0)
                    target_weight = np.concatenate(
                        [target_weight, target_weight_i[None]], axis=0)
            else:
                target, target_weight = self._msra_generate_target(
                    results['ann_info'], joints_3d, joints_3d_visible,
                    self.sigma)

        elif self.encoding == 'Megvii':
            if isinstance(self.kernel, list):
                num_kernels = len(self.kernel)
                cfg = results['ann_info']
                num_joints = cfg['num_joints']
                W, H = cfg['heatmap_size']

                target = np.empty((0, num_joints, H, W), dtype=np.float32)
                target_weight = np.empty((0, num_joints, 1), dtype=np.float32)
                for i in range(num_kernels):
                    target_i, target_weight_i = self._megvii_generate_target(
                        cfg, joints_3d, joints_3d_visible, self.kernel[i])
                    target = np.concatenate([target, target_i[None]], axis=0)
                    target_weight = np.concatenate(
                        [target_weight, target_weight_i[None]], axis=0)
            else:
                target, target_weight = self._megvii_generate_target(
                    results['ann_info'], joints_3d, joints_3d_visible,
                    self.kernel)

        elif self.encoding == 'UDP':
            if self.target_type.lower() == 'CombinedTarget'.lower():
                factors = self.valid_radius_factor
                channel_factor = 3
            elif self.target_type.lower() == 'GaussianHeatmap'.lower():
                factors = self.sigma
                channel_factor = 1
            else:
                raise ValueError('target_type should be either '
                                 "'GaussianHeatmap' or 'CombinedTarget'")
            # if isinstance(factors, list):
            #     num_factors = len(factors)
            #     cfg = results['ann_info']
            #     num_joints = cfg['num_joints']
            #     W, H = cfg['heatmap_size']

            #     target = np.empty((0, channel_factor * num_joints, H, W),
            #                       dtype=np.float32)
            #     target_weight = np.empty((0, num_joints, 1), dtype=np.float32)
            #     targets = []
            #     target_weights = []
            #     for i in range(num_factors):
            #         target_i, target_weight_i = self._udp_generate_target(
            #             cfg, joints_3d, joints_3d_visible, factors[i],
            #             self.target_type, self.valid_visibilities, self.ignore_zeros,
            #             self.normalize, self.probability_map, self.directional_probabilities)
            #         targets.append(target_i[None])
            #         target_weights.append(target_weight_i[None])
            #     target = np.concatenate(targets, axis=0)
            #     target_weight = np.concatenate(target_weights, axis=0)
            # else:
            target, target_weight = self._udp_generate_target(
                results['ann_info'], joints_3d, joints_3d_visible, factors,
                self.target_type, self.valid_visibilities, self.ignore_zeros,
                self.normalize, self.probability_map, self.directional_probabilities)
        else:
            raise ValueError(
                f'Encoding approach {self.encoding} is not supported!')

        if results['ann_info'].get('max_num_joints', None) is not None:
            W, H = results['ann_info']['heatmap_size']
            padded_length = int(results['ann_info'].get('max_num_joints') - results['ann_info'].get('num_joints'))
            target_weight = np.concatenate([target_weight, np.zeros((padded_length, 1), dtype=np.float32)], 0)
            target = np.concatenate([target, np.zeros((padded_length, H, W), dtype=np.float32)], 0)

        results['target'] = target
        results['target_weight'] = target_weight

        results['dataset_idx'] = results['ann_info'].get('dataset_idx', 0)

        return results


@PIPELINES.register_module()
class TopDownGenerateTargetRegression:
    """Generate the target regression vector (coordinates).

    Required keys: 'joints_3d', 'joints_3d_visible', 'ann_info'. Modified keys:
    'target', and 'target_weight'.
    """

    def __init__(self):
        pass

    def _generate_target(self, cfg, joints_3d, joints_3d_visible):
        """Generate the target regression vector.

        Args:
            cfg (dict): data config
            joints_3d: np.ndarray([num_joints, 3])
            joints_3d_visible: np.ndarray([num_joints, 3])

        Returns:
             target, target_weight(1: visible, 0: invisible)
        """
        image_size = cfg['image_size']
        joint_weights = cfg['joint_weights']
        use_different_joint_weights = cfg['use_different_joint_weights']

        mask = (joints_3d[:, 0] >= 0) * (
            joints_3d[:, 0] <= image_size[0] - 1) * (joints_3d[:, 1] >= 0) * (
                joints_3d[:, 1] <= image_size[1] - 1)

        target = joints_3d[:, :2] / image_size

        target = target.astype(np.float32)
        target_weight = joints_3d_visible[:, :2] * mask[:, None]

        if use_different_joint_weights:
            target_weight = np.multiply(target_weight, joint_weights)

        return target, target_weight

    def __call__(self, results):
        """Generate the target heatmap."""
        joints_3d = results['joints_3d']
        joints_3d_visible = results['joints_3d_visible']

        target, target_weight = self._generate_target(results['ann_info'],
                                                      joints_3d,
                                                      joints_3d_visible)

        results['target'] = target
        results['target_weight'] = target_weight

        return results


@PIPELINES.register_module()
class TopDownRandomTranslation:
    """Data augmentation with random translation.

    Required key: 'scale' and 'center'.

    Modifies key: 'center'.

    Note:
        - bbox height: H
        - bbox width: W

    Args:
        trans_factor (float): Translating center to
            ``[-trans_factor, trans_factor] * [W, H] + center``.
        trans_prob (float): Probability of random translation.
    """

    def __init__(self, trans_factor=0.15, trans_prob=1.0):
        self.trans_factor = trans_factor
        self.trans_prob = trans_prob

    def __call__(self, results):
        """Perform data augmentation with random translation."""
        center = results['center']
        scale = results['scale']
        if np.random.rand() <= self.trans_prob:
            # reference bbox size is [200, 200] pixels
            center += self.trans_factor * np.random.uniform(
                -1, 1, size=2) * scale * 200
        results['center'] = center
        return results
