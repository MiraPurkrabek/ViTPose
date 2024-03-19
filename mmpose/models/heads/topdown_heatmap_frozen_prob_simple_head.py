# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, normal_init)
import numpy as np

from mmpose.core.evaluation import pose_pck_accuracy
from mmpose.core.post_processing import flip_back
from mmpose.models.builder import build_loss
from mmpose.models.utils.ops import resize
from ..builder import HEADS
import torch.nn.functional as F
from torch.nn import Sigmoid
from .topdown_heatmap_base_head import TopdownHeatmapBaseHead


@HEADS.register_module()
class TopdownHeatmapFrozenProbSimpleHead(TopdownHeatmapBaseHead):
    """Top-down probability map simple head.

    TopdownProbabilityMapSimpleHead is consisted of (>=0) number of deconv layers
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
                 train_cfg=None,
                 test_cfg=None,
                 upsample=0,
                 normalize=False,
                 use_prelu=False):
        super().__init__()

        self.in_channels = in_channels
        self.loss = build_loss(loss_keypoint)
        self.upsample = upsample

        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        self.target_type = self.test_cfg.get('target_type', 'GaussianHeatmap')

        self._init_inputs(in_channels, in_index, input_transform)
        self.in_index = in_index
        self.align_corners = align_corners

        if use_prelu:
            self.nonlinearity = nn.PReLU()
        else:
            self.nonlinearity = nn.ReLU(inplace=True)

        if extra is not None and not isinstance(extra, dict):
            raise TypeError('extra should be dict or None.')

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
            layers.append(self.nonlinearity)
            
            if len(layers) > 1:
                self.final_layer = nn.Sequential(*layers)
            else:
                self.final_layer = layers[0]

        if normalize:
            self.final_layer = nn.Sequential(self.final_layer, Sigmoid())

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

        for layer in self.deconv_layers:
            layer.requires_grad = False
        for layer in self.final_layer:
            layer.requires_grad = False


    def get_loss(self, output, target, target_weight):
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
        # output = output[:, :, :-1] # Remove the last channel as it is not part of the heatmap
        # output = output.reshape((output.shape[0], output.shape[1], 64, 48))

        # target = target[:, :, :-1] # Remove the last channel as it is not part of the heatmap
        # target = target.reshape((target.shape[0], target.shape[1], 64, 48))

        losses = dict()

        assert not isinstance(self.loss, nn.Sequential)
        assert target.dim() == 3 and target_weight.dim() == 3
        losses['heatmap_loss'] = self.loss(output, target, target_weight)

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
        output = output[:, :, :-1] # Remove the last channel as it is not part of the heatmap
        output = output.reshape((output.shape[0], output.shape[1], 64, 48))

        target = target[:, :, :-1] # Remove the last channel as it is not part of the heatmap
        target = target.reshape((target.shape[0], target.shape[1], 64, 48))

        accuracy = dict()

        if self.target_type == 'GaussianHeatmap':
            _, avg_acc, _ = pose_pck_accuracy(
                output.detach().cpu().numpy(),
                target.detach().cpu().numpy(),
                target_weight.detach().cpu().numpy().squeeze(-1) > 0)
            accuracy['acc_pose'] = float(avg_acc)

        return accuracy

    def forward(self, x):
        """Forward function."""
        x_htm = self.forward_heatmap(x.detach())
        x_pred = self.forward_probability(x)
        
        # Normalize the probability map
        B, C, H, W = x_htm.shape
        x_htm = x_htm.reshape((B, C, -1))
        x_pred = x_pred.reshape((B, C, -1))
        x_all = torch.cat((x_htm, x_pred), dim=2)
        
        return x_all
    
    def forward_heatmap(self, x):
        """Forward fucntion for heatmap prediction."""
        x = x.clone()
        x = self._transform_inputs(x)
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x
    
    def forward_probability(self, x):
        """Forward function for out-of-image probability prediction."""
        y = x.clone()
        y = self.probability_layers(y)
        return y

    def inference_model(self, x, flip_pairs=None):
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
        ooi_prob = output[:, :, -1].detach().cpu().numpy() # Get the out-of-image probability        
        output = output[:, :, :-1] # Remove the last channel as it is not part of the heatmap
        output = output.reshape((B, C, 64, 48))
        
        if flip_pairs is not None:
            output_heatmap = flip_back(
                output.detach().cpu().numpy(),
                flip_pairs,
                target_type=self.target_type)
            # feature is not aligned, shift flipped heatmap for higher accuracy
            if self.test_cfg.get('shift_heatmap', False):
                output_heatmap[:, :, :, 1:] = output_heatmap[:, :, :, :-1]
        else:
            output_heatmap = output.detach().cpu().numpy()
        
        # Normalize the heatmap
        # breakpoint()
        # print("Output heatmap before:", output_heatmap.shape, output_heatmap.min(), output_heatmap.max())
        # for i in range(B):
        #     for j in range(C):
        #         output_heatmap[i, j, :, :] /= 2 * np.pi * 2**2
        # print("Output heatmap after:", output_heatmap.shape, output_heatmap.min(), output_heatmap.max())
        
        # htm_sum = output_heatmap.reshape((B, C, -1)).sum(axis=2) # Get the heatmap sum 
        # mask = (htm_sum > 0) & (ooi_prob < 1)     # Get the mask for the heatmap sum
        # alphas = htm_sum / (1 - ooi_prob) # Calculate the alpha values
        # output_heatmap[mask, :, :] /= alphas[mask, None, None]

        # o = output_heatmap[0, 6, :, :]
        # print()
        # print(o.min(), o.max(), o.sum())
        # print(np.unravel_index(o.argmax(), o.shape))

        # for i in range(B):
        #     for j in range(C):
        #         if alphas[i, j] > 0:
        #             output_heatmap[i, j, :, :] /= alphas[i, j]
        #         elif alphas[i, j] == 0:
        #             output_heatmap[i, j, :, :] /= output_heatmap.sum()
        #         elif alphas[i, j] == 1:
        #             output_heatmap[i, j, :, :] = 0
        # print(alphas[0, 6])
        # o = output_heatmap[0, 6, :, :]
        # print(o.min(), o.max(), o.sum())
        # print(np.unravel_index(o.argmax(), o.shape))
        # breakpoint()

        # new_htm_sum = output_heatmap.reshape(B, C, -1).sum(axis=2)
        # if not (np.all(new_htm_sum >= 0) and np.all(new_htm_sum <= 1)):
        #     breakpoint()

        return output_heatmap

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
