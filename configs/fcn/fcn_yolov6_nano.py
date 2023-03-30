# model settings
_base_ = [
    '../_base_/datasets/cityscapes.py', '../_base_/default_runtime.py',
]
# optimizer
optimizer = dict(type='Adam', lr=1e-3, weight_decay=1e-5)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='CosineAnnealing', warmup='linear',
                 min_lr=1e-6, by_epoch=True, warmup_iters=5, warmup_ratio=0.2)
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=40)
checkpoint_config = dict(by_epoch=True, interval=2)
evaluation = dict(interval=1, metric='mIoU', pre_eval=True)


norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='YOLOv6FPN'),
    decode_head=dict(
        type='ConvHead',
        in_channels=32,
        in_index=0,
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
