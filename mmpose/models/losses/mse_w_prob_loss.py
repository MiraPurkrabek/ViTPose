# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import numpy as np

from ..builder import LOSSES


@LOSSES.register_module()
class JointsMSEProbLoss(nn.Module):
    """MSE loss for heatmaps.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, use_target_weight=False, loss_weight=1.):
        super().__init__()
        self.htm_criterion = nn.MSELoss()
        self.prob_criterion = nn.BCELoss()
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight, return_dict=False):
        """Forward function."""
        batch_size = output.size(0)
        num_joints = output.size(1)

        heatmaps_pred = output.reshape(
            (batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        htm_loss = 0.
        prob_loss = 0.

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(1)  #* 25.5
            heatmap_gt = heatmaps_gt[idx].squeeze(1)      #* 2*np.pi*2**2

            prob_pred = heatmap_pred[:, -1]
            prob_gt = heatmap_gt[:, -1]
            
            heatmap_pred = heatmap_pred[:, :-1]
            heatmap_gt = heatmap_gt[:, :-1]
            
            if self.use_target_weight:
                htm_weight = target_weight[:, idx].clone()
                w = target_weight[:, idx].clone()
            else:
                htm_weight = torch.ones_like(heatmap_gt)
                w = torch.ones_like(prob_gt)
            # For cases where the keypoint is not in the image,
            # train only the probability of OOI, not heatmap
            # htm_weight[prob_gt > 0.99] = 0 

            weighted_htm_pred = heatmap_pred * htm_weight
            weighted_htm_gt = heatmap_gt * htm_weight
            weighted_prob_pred = prob_pred * w
            weighted_prob_gt = prob_gt * w

            htm_loss += self.htm_criterion(weighted_htm_pred, weighted_htm_gt)
            prob_loss += self.prob_criterion(weighted_prob_pred, weighted_prob_gt)

        htm_loss = htm_loss / num_joints * self.loss_weight
        prob_loss = prob_loss / num_joints * self.loss_weight# * 1e-3


        if return_dict:
            loss = dict(heatmap_loss=htm_loss, probability_loss=prob_loss)
        else:
            loss = htm_loss + prob_loss

        return loss
