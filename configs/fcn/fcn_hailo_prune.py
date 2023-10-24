# model settings
_base_ = [
    '../_base_/datasets/cityscapes10classes.py', '../_base_/default_runtime.py',
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
train_cfg = dict(type='IterBasedTrainLoop', max_iters=104160, val_interval=1488)  # 74400 (50 epochs), 89280 (60 epochs), 104160 (70 epochs), 119040 (80 epochs)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# default hooks - logger & checkpoint configs
default_hooks = dict(

    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=200, log_metric_by_epoch=False),

    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),

    # save checkpoint every 2 epochs.
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2976),  # 7440 (5 Epoches)
)

# custom hooks
custom_hooks = [dict(type='SparseMLHook', interval=10, priority='NORMAL')]

# tensorboard vis
vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend', draw=True, interval=10, save_dir='tf_dir')]  # , draw=True, interval=1

# data preprocessing
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (512, 1024)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[0.0, 0.0, 0.0],
    std=[1.0, 1.0, 1.0],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='hailoFPN',
        depth=0.33,
        width=0.125,
        bb_channels_list=[128, 256, 512, 1024],
        bb_num_repeats_list=[9, 15, 21, 12],
        neck_channels_list=[256, 128, 128, 256, 256, 512],
        neck_num_repeats_list=[9, 12, 12, 9]),
    decode_head=dict(
        type='ConvHead',
        in_channels=16,
        channels=128,
        num_convs=1,
        num_classes=10,
        norm_cfg=norm_cfg,
        align_corners=True,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
    infer_wo_softmax=True)

resume = True
load_from='./fcn_hailo_10classes_sholev.pth'
