# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import numpy as np

from ..builder import LOSSES


@LOSSES.register_module()
class KLDloss(nn.Module):
    """Kullback-Leibler divergence loss.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, use_target_weight=False, loss_weight=1.):
        super().__init__()
        self.criterion = nn.KLDivLoss(log_target=False, reduction='batchmean')
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight):
        """Forward function."""
        batch_size = output.size(0)
        num_joints = output.size(1)

        heatmaps_pred = output.reshape(
            (batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = 0.
        # print("="*30)

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(1)
            heatmap_gt = heatmaps_gt[idx].squeeze(1)

            # Add laplacian noise and normalize
            heatmap_pred = heatmap_pred + 1e-12
            heatmap_gt = heatmap_gt + 1e-12
            heatmap_pred = nn.functional.normalize(heatmap_pred, p=1)
            heatmap_gt = nn.functional.normalize(heatmap_gt, p=1)

            # Make the prediction in log-space as required by PyTorch
            heatmap_pred = heatmap_pred.log()

            if self.use_target_weight:
                l = self.criterion(heatmap_pred * target_weight[:, idx],
                                       heatmap_gt * target_weight[:, idx])
            else:
                l = self.criterion(heatmap_pred, heatmap_gt)
        
        loss += l

        # print("---")
        # print(loss)

        return loss / num_joints * self.loss_weight
