# model settings
_base_ = [
    '../_base_/datasets/cityscapes10classes.py', '../_base_/default_runtime.py',
]

resume = True
# best checkpoint path of full training (fcn_hailo_10classes). Start of pruning procedure:
load_from = './work_dirs/fcn_hailo_eta1e5_eve/iter_74400.pth'

# optimizer
optimizer = dict(type='Adam', lr=0.0001, weight_decay=1e-5)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)


# runtime settings
train_cfg = dict(type='IterBasedTrainLoop', max_iters=178560, val_interval=1488)  # 74400 (50 epochs), 178560 (120)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# default hooks - logger & checkpoint configs
default_hooks = dict(

    # print log every 500 iterations.
    logger=dict(type='LoggerHook', interval=500, log_metric_by_epoch=False),

    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),
    )

# learning policy: taken from the recipe
# custom hooks
sparseml_hook = dict(type='SparseMLHook', priority='NORMAL')
# Saving best checkpoint starts after pruning hits final ratio
ext_checkpoint_hook = dict(type='ExtCheckpointHook', by_epoch=False, interval=1488, save_best='mIoU', rule='greater',
                           max_keep_ckpts=5, save_begin=163680)  # 163680 (110 epochs)
custom_hooks = [sparseml_hook, ext_checkpoint_hook]

# tensorboard vis ('LocalVisBackend' might be redundant)  save_dir='./tf_dir/<exp_name>'
visualizer = dict(type='SegLocalVisualizer',
                  vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')],
                  name='visualizer')

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
