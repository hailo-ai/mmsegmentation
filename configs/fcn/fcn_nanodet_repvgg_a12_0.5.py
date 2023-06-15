# model settings
_base_ = [
    '../_base_/datasets/cityscapes.py', '../_base_/default_runtime.py',
]

# optimizer
optimizer = dict(type='Adam', lr=0.001, weight_decay=1e-5)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

# learning policy
param_scheduler = [
	dict(
		type='LinearLR', start_factor=0.2, by_epoch=False, begin=0, end=7440),
    dict(
        type='CosineAnnealingLR', begin=7440, by_epoch=False, end=59520)
]

# runtime settings
train_cfg = dict(type='IterBasedTrainLoop', max_iters=59520, val_interval=1488)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# default hooks - logger & checkpoint configs
default_hooks = dict(

    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),

    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),

    # save checkpoint every 5 epochs.
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=7440),
)

# tensorboard vis
vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]

# data preprocessing
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (512, 1024)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

fpn_cfg =dict(
    name="PAN",
    in_channels=[64, 64, 256],
    out_channels=128,
    start_level=0,
    num_outs=3,
)

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='RepVGG',
        num_blocks=[2, 4, 14, 1],
        width_multiplier=[0.5, 0.5, 0.25, 1.25], 
        override_groups_map=None, deploy=False, fpn_cfg=fpn_cfg),
    decode_head=dict(
        type='ConvHeadx4',
        in_channels=128,
        in_index=0,  # nanodet_repvgg backbone outputs = [batch, 128, 80, 80], [batch, 128, 40, 40], [batch, 128, 20, 20] - this selects [batch, 128, 40, 40], [batch, 128, 20, 20]  for the decode head
        channels=128,
        num_convs=1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=True,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
    infer_wo_softmax=True)