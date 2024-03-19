# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import numpy as np

from ..builder import LOSSES


@LOSSES.register_module()
class JointsMSELoss(nn.Module):
    """MSE loss for heatmaps.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, use_target_weight=False, loss_weight=1.):
        super().__init__()
        self.criterion = nn.MSELoss()
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight, reduction='mean'):
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

            if self.use_target_weight:
                l = self.criterion(heatmap_pred * target_weight[:, idx],
                                       heatmap_gt * target_weight[:, idx])
            else:
                l = self.criterion(heatmap_pred, heatmap_gt)
            
            loss += l
            
        loss = loss / num_joints * self.loss_weight

        return loss


@LOSSES.register_module()
class JointsNormalizedMSELoss(nn.Module):
    """Normalized MSE loss for heatmaps.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, use_target_weight=False, loss_weight=1.):
        super().__init__()
        self.shape_criterion = nn.MSELoss()
        self.pbt_criterion = nn.CrossEntropyLoss()
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight
        self.sigmoid = nn.Sigmoid()

    def forward(self, output, target, target_weight):
        """Forward function."""
        batch_size = output.size(0)
        num_joints = output.size(1)

        print("Output", output.shape, output.min(), output.max())
        print("Target", target.shape, target.min(), target.max())

        heatmaps_pred = output.reshape(
            (batch_size, num_joints, -1))#.split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1))#.split(1, 1)

        print("Pred heatmaps", heatmaps_pred.shape, heatmaps_pred.min(), heatmaps_pred.max())
        print("GT   heatmaps", heatmaps_gt.shape, heatmaps_gt.min(), heatmaps_gt.max())
        print("Target weights", target_weight.shape)
        
        # Get probability of joint presence
        probabilities_pred = self.sigmoid(torch.sum(heatmaps_pred, dim=2))
        probabilities_gt = self.sigmoid(torch.sum(heatmaps_gt, dim=2))
        
        print("Pred probabilities", probabilities_pred.shape, probabilities_pred.min(), probabilities_pred.max())
        print("GT   probabilities", probabilities_gt.shape, probabilities_gt.min(), probabilities_gt.max())

        # Normalize heatmaps between 0 and 1
        heatmaps_pred -= heatmaps_pred.min()
        heatmaps_pred /= heatmaps_pred.max()
        heatmaps_gt -= heatmaps_gt.min()
        heatmaps_gt /= heatmaps_gt.max()

        print("Pred norm heatmaps", heatmaps_pred.shape, heatmaps_pred.min(), heatmaps_pred.max())
        print("GT   nrom heatmaps", heatmaps_gt.shape, heatmaps_gt.min(), heatmaps_gt.max())

        # Split values by kpt_idx
        heatmaps_pred = heatmaps_pred.split(1, 1)
        heatmaps_gt = heatmaps_gt.split(1, 1)
        probabilities_gt = probabilities_gt.split(1, 1)
        probabilities_pred = probabilities_pred.split(1, 1)

        print("Pred heatmaps", len(heatmaps_pred), heatmaps_pred[0].shape)
        print("GT   heatmaps", len(heatmaps_gt), heatmaps_gt[0].shape)
        print("Pred probabilities", len(probabilities_pred), probabilities_pred[0].shape)
        print("GT   probabilities", len(probabilities_gt), probabilities_gt[0].shape)

        # heatmaps_gt_pbt = np.array([torch.sum(heatmaps_gt[i], dim=2) for i in range(num_joints)])
        # heatmaps_pred_pbt = np.array([torch.sum(heatmaps_pred[i], dim=2) for i in range(num_joints)])

        # print("Pred heatmaps pbt", heatmaps_pred_pbt.shape, heatmaps_pred_pbt.min(), heatmaps_pred_pbt.max())
        # print("GT   heatmaps pbt", heatmaps_gt_pbt.shape, heatmaps_gt_pbt.min(), heatmaps_gt_pbt.max())


        loss = 0.

        # Check if pred is still grad enabled
        print("Pred heatmaps", heatmaps_pred[0].requires_grad)

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(1)
            heatmap_gt = heatmaps_gt[idx].squeeze(1)
            print("Pred heatmap", heatmap_pred.requires_grad)
            if self.use_target_weight:
                loss += self.shape_criterion(heatmap_pred * target_weight[:, idx],
                                       heatmap_gt * target_weight[:, idx])
                loss += self.pbt_criterion(probabilities_pred[idx] * target_weight[:, idx],
                                           probabilities_gt[idx] * target_weight[:, idx])
            else:
                loss += self.shape_criterion(heatmap_pred, heatmap_gt)
                loss += self.pbt_criterion(probabilities_pred[idx], probabilities_gt[idx])
            print("loss", loss.requires_grad)

        # raise NotImplementedError
        return loss / num_joints * self.loss_weight


@LOSSES.register_module()
class CombinedTargetMSELoss(nn.Module):
    """MSE loss for combined target.
        CombinedTarget: The combination of classification target
        (response map) and regression target (offset map).
        Paper ref: Huang et al. The Devil is in the Details: Delving into
        Unbiased Data Processing for Human Pose Estimation (CVPR 2020).

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, use_target_weight, loss_weight=1.):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_channels = output.size(1)
        heatmaps_pred = output.reshape(
            (batch_size, num_channels, -1)).split(1, 1)
        heatmaps_gt = target.reshape(
            (batch_size, num_channels, -1)).split(1, 1)
        loss = 0.
        num_joints = num_channels // 3
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx * 3].squeeze()
            heatmap_gt = heatmaps_gt[idx * 3].squeeze()
            offset_x_pred = heatmaps_pred[idx * 3 + 1].squeeze()
            offset_x_gt = heatmaps_gt[idx * 3 + 1].squeeze()
            offset_y_pred = heatmaps_pred[idx * 3 + 2].squeeze()
            offset_y_gt = heatmaps_gt[idx * 3 + 2].squeeze()
            if self.use_target_weight:
                heatmap_pred = heatmap_pred * target_weight[:, idx]
                heatmap_gt = heatmap_gt * target_weight[:, idx]
            # classification loss
            loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
            # regression loss
            loss += 0.5 * self.criterion(heatmap_gt * offset_x_pred,
                                         heatmap_gt * offset_x_gt)
            loss += 0.5 * self.criterion(heatmap_gt * offset_y_pred,
                                         heatmap_gt * offset_y_gt)
        return loss / num_joints * self.loss_weight


@LOSSES.register_module()
class JointsOHKMMSELoss(nn.Module):
    """MSE loss with online hard keypoint mining.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        topk (int): Only top k joint losses are kept.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, use_target_weight=False, topk=8, loss_weight=1.):
        super().__init__()
        assert topk > 0
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk
        self.loss_weight = loss_weight

    def _ohkm(self, loss):
        """Online hard keypoint mining."""
        ohkm_loss = 0.
        N = len(loss)
        for i in range(N):
            sub_loss = loss[i]
            _, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False)
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= N
        return ohkm_loss

    def forward(self, output, target, target_weight):
        """Forward function."""
        batch_size = output.size(0)
        num_joints = output.size(1)
        if num_joints < self.topk:
            raise ValueError(f'topk ({self.topk}) should not '
                             f'larger than num_joints ({num_joints}).')
        heatmaps_pred = output.reshape(
            (batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        losses = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(1)
            heatmap_gt = heatmaps_gt[idx].squeeze(1)
            if self.use_target_weight:
                losses.append(
                    self.criterion(heatmap_pred * target_weight[:, idx],
                                   heatmap_gt * target_weight[:, idx]))
            else:
                losses.append(self.criterion(heatmap_pred, heatmap_gt))

        losses = [loss.mean(dim=1).unsqueeze(dim=1) for loss in losses]
        losses = torch.cat(losses, dim=1)

        return self._ohkm(losses) * self.loss_weight
