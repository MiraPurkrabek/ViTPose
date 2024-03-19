# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import numpy as np

from ..builder import LOSSES



@LOSSES.register_module()
class BCEloss(nn.Module):
    """Jensen-Shannon Divergence loss.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, use_target_weight=False, loss_weight=1.):
        super().__init__()
        self.criterion = nn.BCELoss()
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

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(1)
            heatmap_gt = heatmaps_gt[idx].squeeze(1)

            prob_pred = heatmap_pred[:, -1]
            prob_gt = heatmap_gt[:, -1]
            
            if self.use_target_weight:
                l = self.criterion(prob_pred * target_weight[:, idx],
                                       prob_gt * target_weight[:, idx])
            else:
                l = self.criterion(prob_pred, prob_gt)
            loss += l

        return loss / num_joints * self.loss_weight
