# Copyright (c) OpenMMLab. All rights reserved.
from .bce_loss import BCEloss
from .ce_loss import CEloss
from .classfication_loss import BCELoss
from .heatmap_loss import AdaptiveWingLoss
from .jsd_loss import JSDloss
from .kld_loss import KLDloss
from .mesh_loss import GANLoss, MeshLoss
from .mse_loss import JointsMSELoss, JointsOHKMMSELoss, JointsNormalizedMSELoss
from .mse_w_prob_loss import JointsMSEProbLoss
from .multi_loss_factory import AELoss, HeatmapLoss, MultiLossFactory
from .regression_loss import (BoneLoss, L1Loss, MPJPELoss, MSELoss,
                              SemiSupervisionLoss, SmoothL1Loss, SoftWingLoss,
                              WingLoss, L1LogLoss)

__all__ = [
    'JointsMSELoss', 'JointsOHKMMSELoss', 'HeatmapLoss', 'AELoss',
    'MultiLossFactory', 'MeshLoss', 'GANLoss', 'SmoothL1Loss', 'WingLoss',
    'MPJPELoss', 'MSELoss', 'L1Loss', 'BCELoss', 'BoneLoss',
    'SemiSupervisionLoss', 'SoftWingLoss', 'AdaptiveWingLoss',
    'JointsNormalizedMSELoss', 'KLDloss', 'JSDloss', 'JointsMSEProbLoss',
    'CEloss', 'BCEloss', 'L1LogLoss'
]
