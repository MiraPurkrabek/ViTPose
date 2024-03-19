# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, normal_init)
import numpy as np
import sklearn.metrics as metrics


from mmpose.core.evaluation import pose_pck_accuracy, _get_max_preds, post_dark_udp
from mmpose.core.post_processing import flip_back
from mmpose.models.builder import build_loss
from mmpose.models.utils.ops import resize
from ..builder import HEADS
import torch.nn.functional as F
from torch.nn import Sigmoid
from .topdown_heatmap_base_head import TopdownHeatmapBaseHead


@HEADS.register_module()
class TopdownHeatmapFullHeadFromHTM(TopdownHeatmapBaseHead):
    """Top-down heatmap full head. paper ref: Bin Xiao et al. ``Simple
    Baselines for Human Pose Estimation and Tracking``.

    TopdownHeatmapSimpleHead is consisted of (>=0) number of deconv layers
    and a simple conv2d layer.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        num_deconv_layers (int): Number of deconv layers.
            num_deconv_layers should >= 0. Note that 0 means
            no deconv layers.
        num_deconv_filters (list|tuple): Number of filters.
            If num_deconv_layers > 0, the length of
        num_deconv_kernels (list|tuple): Kernel sizes.
        in_index (int|Sequence[int]): Input feature index. Default: 0
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            Default: None.

            - 'resize_concat': Multiple feature maps will be resized to the
                same size as the first one and then concat together.
                Usually used in FCN head of HRNet.
            - 'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            - None: Only one select feature map is allowed.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        loss_keypoint (dict): Config for keypoint loss. Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_deconv_layers=3,
                 num_deconv_filters=(256, 256, 256),
                 num_deconv_kernels=(4, 4, 4),
                 extra=None,
                 in_index=0,
                 input_transform=None,
                 align_corners=False,
                 loss_keypoint=None,
                 loss_probability=None,
                 loss_error=None,
                 train_cfg=None,
                 test_cfg=None,
                 upsample=0,
                 normalize=False,
                 use_prelu=False,
                 freeze_localization_head=False,
                 freeze_probability_head=False,
                 freeze_error_head=False,):
        super().__init__()

        self.in_channels = in_channels
        self.keypoint_loss = build_loss(loss_keypoint)

        if loss_probability is not None:
            self.probability_loss = build_loss(loss_probability)
        else:
            self.probability_loss = None
        
        if loss_error is not None:
            self.error_loss = build_loss(loss_error)
        else:
            self.error_loss = None

        self.upsample = upsample

        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        self.target_type = self.test_cfg.get('target_type', 'GaussianHeatmap')

        self._init_inputs(in_channels, in_index, input_transform)
        self.in_index = in_index
        self.align_corners = align_corners

        if use_prelu:
            self.nonlinearity = nn.PReLU()
            # self.nonlinearity = nn.ReLU(inplace=True)
        else:
            self.nonlinearity = nn.ReLU(inplace=True)

        if extra is not None and not isinstance(extra, dict):
            raise TypeError('extra should be dict or None.')
        
        self._build_localication_head(
            out_channels, num_deconv_layers, num_deconv_filters, num_deconv_kernels,
            extra, normalize, freeze_localization_head
        )

        self._build_probability_head(
            in_channels, out_channels, freeze_probability_head
        )

        self._build_error_head(
            in_channels, out_channels, freeze_error_head
        )


    def _build_localication_head(
        self, 
        out_channels,
        num_deconv_layers,
        num_deconv_filters,
        num_deconv_kernels,
        extra,
        normalize,
        freeze_localization_head
    ):
        if num_deconv_layers > 0:
            self.deconv_layers = self._make_deconv_layer(
                num_deconv_layers,
                num_deconv_filters,
                num_deconv_kernels,
            )
        elif num_deconv_layers == 0:
            self.deconv_layers = nn.Identity()
        else:
            raise ValueError(
                f'num_deconv_layers ({num_deconv_layers}) should >= 0.')

        identity_final_layer = False
        if extra is not None and 'final_conv_kernel' in extra:
            assert extra['final_conv_kernel'] in [0, 1, 3]
            if extra['final_conv_kernel'] == 3:
                padding = 1
            elif extra['final_conv_kernel'] == 1:
                padding = 0
            else:
                # 0 for Identity mapping.
                identity_final_layer = True
            kernel_size = extra['final_conv_kernel']
        else:
            kernel_size = 1
            padding = 0

        if identity_final_layer:
            self.final_layer = nn.Identity()
        else:
            conv_channels = num_deconv_filters[
                -1] if num_deconv_layers > 0 else self.in_channels

            layers = []
            if extra is not None:
                num_conv_layers = extra.get('num_conv_layers', 0)
                num_conv_kernels = extra.get('num_conv_kernels',
                                             [1] * num_conv_layers)

                for i in range(num_conv_layers):
                    layers.append(
                        build_conv_layer(
                            dict(type='Conv2d'),
                            in_channels=conv_channels,
                            out_channels=conv_channels,
                            kernel_size=num_conv_kernels[i],
                            stride=1,
                            padding=(num_conv_kernels[i] - 1) // 2))
                    layers.append(
                        build_norm_layer(dict(type='BN'), conv_channels)[1])
                    layers.append(self.nonlinearity)

            layers.append(
                build_conv_layer(
                    cfg=dict(type='Conv2d'),
                    in_channels=conv_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding))
            
            if len(layers) > 1:
                self.final_layer = nn.Sequential(*layers)
            else:
                self.final_layer = layers[0]

        if normalize:
            self.final_layer = nn.Sequential(self.final_layer, Sigmoid())

        if freeze_localization_head:
            for param in self.deconv_layers.parameters():
                param.requires_grad = False
            for param in self.final_layer.parameters():
                param.requires_grad = False


    def _build_probability_head(
        self, in_channels, out_channels, freeze_probability_head,      
    ):
        ppb_layers = []
        kernel_sizes = [(4, 3), (2, 2), (2, 2)]
        for i in range(len(kernel_sizes)):
            ppb_layers.append(
                build_conv_layer(
                    dict(type='Conv2d'),
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1))
            ppb_layers.append(
                build_norm_layer(dict(type='BN'), in_channels)[1])
            ppb_layers.append(
                nn.MaxPool2d(kernel_size=kernel_sizes[i], stride=kernel_sizes[i], padding=0))
            ppb_layers.append(self.nonlinearity)
        ppb_layers.append(
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=384,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0))
        ppb_layers.append(nn.Sigmoid())
        self.probability_layers = nn.Sequential(*ppb_layers)

        if freeze_probability_head:
            for param in self.probability_layers.parameters():
                param.requires_grad = False


    def _build_error_head(
        self, in_channels, out_channels, freeze_error_head,
    ):
        err_layers = []
            
        kernel_sizes = [(4, 3), (4, 4), (4, 4)]
        for i in range(len(kernel_sizes)):
            err_layers.append(
                build_conv_layer(
                    dict(type='Conv2d'),
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1))
            err_layers.append(
                build_norm_layer(dict(type='BN'), out_channels)[1])
            err_layers.append(
                nn.MaxPool2d(kernel_size=kernel_sizes[i], stride=kernel_sizes[i], padding=0))
            err_layers.append(self.nonlinearity)
        err_layers.append(
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0))
        err_layers.append(self.nonlinearity)
        self.error_layers = nn.Sequential(*err_layers)

        if freeze_error_head:
            for param in self.error_layers.parameters():
                param.requires_grad = False


    def _error_from_heatmaps(self, pred_heatmaps, target_heatmaps):
        if not isinstance(pred_heatmaps, np.ndarray):
            pred_heatmaps = pred_heatmaps.detach().clone().cpu().numpy()
        if not isinstance(target_heatmaps, np.ndarray):
            target_heatmaps = target_heatmaps.detach().clone().cpu().numpy()
        
        # Get the GT location from target heatmap
        gt_coords_coarse, maxvals = _get_max_preds(target_heatmaps)

        # Get the estimated location from the output heatmap
        dt_coords_coarse, _ = _get_max_preds(pred_heatmaps)

        # Improve localization with DARK
        gt_coords = post_dark_udp(gt_coords_coarse, target_heatmaps)
        dt_coords = post_dark_udp(dt_coords_coarse, pred_heatmaps)
        
        # Calculate the error
        target_errors = np.linalg.norm(gt_coords - dt_coords, axis=2)
        assert (target_errors >= 0).all(), "Euclidean distance cannot be negative"

        return target_errors


    def get_loss(self, output, target, target_weight, reduction='mean'):
        """Calculate top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (torch.Tensor[N,K,H,W]): Output heatmaps.
            target (torch.Tensor[N,K,H,W]): Target heatmaps.
            target_weight (torch.Tensor[N,K,1]):
                Weights across different joint types.
        """

        # Extract heatmap, probability and error from output
        N, K, _ = output.shape
        pred_heatmaps = output[:, :, :-2]
        pred_heatmaps = pred_heatmaps.reshape(N, K, 64, 48)
        pred_probs = output[:, :, -2].unsqueeze(-1)
        pred_errors = output[:, :, -1].unsqueeze(-1)

        # Extract heatmap and probability from target
        target_heatmaps = target[:, :, :-1]
        target_heatmaps = target_heatmaps.reshape(N, K, 64, 48)
        target_probs = target[:, :, -1].unsqueeze(-1)

        # Calculate the error from the heatmaps
        target_errors = self._error_from_heatmaps(pred_heatmaps, target_heatmaps)
        target_errors = torch.tensor(target_errors).float().cuda()        
        target_errors = target_errors.unsqueeze(-1)
        target_errors = target_errors * (target_weight > 0)

        losses = dict()

        assert not isinstance(self.keypoint_loss, nn.Sequential)
        assert target_heatmaps.dim() == 4 and target_weight.dim() == 3
        losses['heatmap_loss'] = self.keypoint_loss(pred_heatmaps, target_heatmaps, target_weight)

        if self.probability_loss is not None:
            losses['probability_loss'] = self.probability_loss(pred_probs, target_probs, target_weight)

        if self.error_loss is not None:
            losses['error_loss'] = self.error_loss(pred_errors, target_errors, target_weight)

        return losses


    def get_accuracy(self, output, target, target_weight):
        """Calculate accuracy for top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (torch.Tensor[N,K,H,W]): Output heatmaps.
            target (torch.Tensor[N,K,H,W]): Target heatmaps.
            target_weight (torch.Tensor[N,K,1]):
                Weights across different joint types.
        """

        accuracy = dict()

        # Extract heatmap, probability and error from output
        N, K, _ = output.shape
        pred_heatmaps = output[:, :, :-2]
        pred_heatmaps = pred_heatmaps.reshape(N, K, 64, 48)
        pred_probs = output[:, :, -2].unsqueeze(-1)
        pred_errors = output[:, :, -1].unsqueeze(-1)

        # Extract heatmap and probability from target
        target_heatmaps = target[:, :, :-1]
        target_heatmaps = target_heatmaps.reshape(N, K, 64, 48)
        target_probs = target[:, :, -1].unsqueeze(-1)

        # Make everything NumPy
        pred_heatmaps = pred_heatmaps.detach().cpu().numpy()
        pred_probs = pred_probs.detach().cpu().numpy()
        pred_errors = pred_errors.detach().cpu().numpy()
        target_heatmaps = target_heatmaps.detach().cpu().numpy()
        target_probs = target_probs.detach().cpu().numpy()
        target_weight = target_weight.detach().cpu().numpy().squeeze(-1)

        # Calculate the error from the heatmaps
        target_errors = self._error_from_heatmaps(pred_heatmaps, target_heatmaps)
        target_errors = target_errors * target_weight

        if self.target_type == 'GaussianHeatmap':
            # Calculate pose accuracy (= localization)
            _, avg_acc, _ = pose_pck_accuracy(
                pred_heatmaps,
                target_heatmaps,
                target_weight > 0)
            accuracy['acc_pose'] = float(avg_acc)

            # Calculate probability accuracy
            prob_roc_auc = metrics.roc_auc_score(target_probs.flatten(), pred_probs.flatten())
            accuracy['prob_roc_auc'] = float(prob_roc_auc)

            # Calculate accuracy when confidence is considered as probability
            conf_probs = np.max(pred_heatmaps, axis=(2, 3)).squeeze()
            conf_roc_auc = metrics.roc_auc_score(target_probs.flatten(), conf_probs.flatten())
            accuracy['conf_roc_auc'] = float(conf_roc_auc)

            # Calculate error accuracy
            error_mae = np.abs(
                target_errors.squeeze() - pred_errors.squeeze()
            )
            error_mae = (error_mae * (target_weight > 0)).mean()
            accuracy['error_mae'] = float(error_mae)
            accuracy['error_mae_norm'] = float(error_mae / np.sqrt(64**2 + 48**2))

        return accuracy


    def forward(self, x):
        """Forward function."""
        # ToDo
        x_htm = self.forward_localization_head(x)
        x_prob = self.forward_probability_head(x)
        # x_prob = self.forward_error_head(x_htm)
        x_err = self.forward_error_head(x_htm)

        # Flatten and concatenate the output
        B, C, H, W = x_htm.shape
        x_htm = x_htm.reshape(B, C, -1)
        x_prob = x_prob.reshape(B, C, -1)
        x_err = x_err.reshape(B, C, -1)

        x_out = torch.cat((x_htm, x_prob, x_err), dim=2)

        return x_out


    def forward_localization_head(self, x):
        """Forward function for localization head."""
        x = x.clone()
        x = self._transform_inputs(x)
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x


    def forward_probability_head(self, x):
        """Forward function for probability head."""
        x = x.clone()
        x = self.probability_layers(x)
        return x


    def forward_error_head(self, x):
        """Forward function for error head."""
        x = x.detach().clone()
        x = self.error_layers(x)
        return x


    def inference_model(self, x, flip_pairs=None, legacy=False, **kwargs):
        """Inference function.

        Returns:
            output_heatmap (np.ndarray): Output heatmaps.

        Args:
            x (torch.Tensor[N,K,H,W]): Input features.
            flip_pairs (None | list[tuple]):
                Pairs of keypoints which are mirrored.
        """
        output = self.forward(x)
        B, C, _ = output.shape
        pred_heatmaps = output[:, :, :-2]
        pred_heatmaps = pred_heatmaps.reshape(B, C, 64, 48)
        pred_heatmaps = pred_heatmaps.detach().cpu().numpy()
        pred_probs = output[:, :, -2].detach().cpu().numpy()
        pred_errors = output[:, :, -1].detach().cpu().numpy()
        pred_errors = pred_errors / np.sqrt(64**2 + 48**2)

        if flip_pairs is not None:
            pred_heatmaps = flip_back(
                pred_heatmaps,
                flip_pairs,
                target_type=self.target_type)
            
            # feature is not aligned, shift flipped heatmap for higher accuracy
            if self.test_cfg.get('shift_heatmap', False):
                pred_heatmaps[:, :, :, 1:] = pred_heatmaps[:, :, :, :-1]
        
            # Flip back probabilities and errors
            pred_probs_flipped = pred_probs.copy()
            pred_errors_flipped = pred_errors.copy()
            for left, right in flip_pairs:
                pred_probs_flipped[:, [left, right]] = pred_probs_flipped[:, [right, left]]
                pred_errors_flipped[:, [left, right]] = pred_errors_flipped[:, [right, left]]
            pred_probs = pred_probs_flipped
            pred_errors = pred_errors_flipped

        if legacy:
            return pred_heatmaps
        else:
            return pred_heatmaps, pred_probs, pred_errors


    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform is not None, in_channels and in_index must be
        list or tuple, with the same length.

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.

                - 'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                - 'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                - None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels


    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor] | Tensor): multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """
        if not isinstance(inputs, list):
            if not isinstance(inputs, list):
                if self.upsample > 0:
                    inputs = resize(
                        input=F.relu(inputs),
                        scale_factor=self.upsample,
                        mode='bilinear',
                        align_corners=self.align_corners
                        )
            return inputs

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs


    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        if num_layers != len(num_filters):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_filters({len(num_filters)})'
            raise ValueError(error_msg)
        if num_layers != len(num_kernels):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_kernels({len(num_kernels)})'
            raise ValueError(error_msg)

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=self.in_channels,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(self.nonlinearity)
            self.in_channels = planes

        return nn.Sequential(*layers)


    def init_weights(self):
        """Initialize model weights."""
        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.PReLU):
                constant_init(m, 1)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.PReLU):
                constant_init(m, 1)
