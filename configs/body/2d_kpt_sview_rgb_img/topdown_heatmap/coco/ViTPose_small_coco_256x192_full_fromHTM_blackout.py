COCO_ROOT = '/datagrid/personal/purkrmir/data/COCO/original'
# COCO_ROOT = '/datagrid/personal/purkrmir/data/OOI_eval/coco_cropped_v2/'

# VAL_COCO_ROOT = '/datagrid/personal/purkrmir/data/OOI_eval/coco_cropped_v2/'
VAL_COCO_ROOT = COCO_ROOT


BATCH_SIZE = 64
PADDING = 1.25

prtr = None
load_from = "models/my/reproduce_epoch_205.pth"

_base_ = [
    '../../../../_base_/default_runtime.py',
    '../../../../_base_/datasets/coco.py'
]
evaluation = dict(interval=1, metric='mAP', save_best='AP')

optimizer = dict(type='AdamW', lr=5e-4, betas=(0.9, 0.999), weight_decay=0.1,
                 constructor='LayerDecayOptimizerConstructor', 
                 paramwise_cfg=dict(
                                    num_layers=12, 
                                    layer_decay_rate=0.8,
                                    custom_keys={
                                            'bias': dict(decay_multi=0.),
                                            'pos_embed': dict(decay_mult=0.),
                                            'relative_position_bias_table': dict(decay_mult=0.),
                                            'norm': dict(decay_mult=0.)
                                            }
                                    )
                )

optimizer_config = dict(grad_clip=dict(max_norm=1., norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[35, 45])
total_epochs = 50
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
target_type = 'GaussianHeatmap'
channel_cfg = dict(
    num_output_channels=17,
    dataset_joints=17,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ])

# model settings
model = dict(
    type='TopDown',
    pretrained=prtr,
    backbone=dict(
        type='ViT',
        img_size=(256, 192),
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=12,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.1,
        frozen_stages=11,
        freeze_attn=True,
        freeze_ffn=True,
    ),
    keypoint_head=dict(
        type='TopdownHeatmapFullHeadFromHTM',
        in_channels=384,
        out_channels=channel_cfg['num_output_channels'],
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(final_conv_kernel=1, ),
        in_index=0,
        input_transform=None,
        align_corners=False,
        upsample=0,
        train_cfg=None,
        test_cfg=None,
        
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True),
        loss_probability=dict(type='BCELoss', use_target_weight=False),
        loss_error=dict(type='L1LogLoss', use_target_weight=True),
        normalize=True,
        use_prelu=False,
        freeze_localization_head=False,
        freeze_probability_head=False,
        freeze_error_head=False,
    ),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=False,
        target_type=target_type,
        modulate_kernel=11,
        use_udp=True))

data_cfg = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    # bbox_file=VAL_COCO_ROOT + "/annotations/coco_val_perfect_dets.json",
    bbox_file=VAL_COCO_ROOT + "/annotations/person_keypoints_val2017.json",
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=8,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.3),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='RandomBlackMask', mask_prob=0.9, min_mask=0.1, max_mask=0.3),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='TopDownGenerateTarget',
        sigma=2,
        encoding='UDP',
        target_type=target_type,
        normalize=False,
        probability_map=True,
        ignore_zeros=True),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs', 'orig_joints_3d', 'joints_3d_visible',
        ]),
]

test_pipeline = val_pipeline

data_root = COCO_ROOT
val_data_root = VAL_COCO_ROOT
data = dict(
    samples_per_gpu=BATCH_SIZE,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=BATCH_SIZE),
    test_dataloader=dict(samples_per_gpu=BATCH_SIZE),
    train=dict(
        type='TopDownCocoDataset',
        ann_file=f'{data_root}/annotations/person_keypoints_train2017.json',
        img_prefix=f'{data_root}/train2017/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='TopDownCocoDataset',
        ann_file=f'{VAL_COCO_ROOT}/annotations/person_keypoints_val2017.json',
        img_prefix=f'{VAL_COCO_ROOT}/val2017/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='TopDownCocoDataset',
        ann_file=f'{VAL_COCO_ROOT}/annotations/person_keypoints_val2017.json',
        img_prefix=f'{VAL_COCO_ROOT}/val2017/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)

