_base_ = [
    '../_base_/models/upernet_vit_adapter.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
# pretrained = 'https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth'  # noqa: E501
pretrained = 'pretrained/deit_tiny_patch16_224-a1311bcf.pth'
model = dict(
    pretrained=pretrained,
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)))
# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=12e-5,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.95))
lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

checkpoint_config = dict(interval=1000, max_keep_ckpts=1)
evaluation = dict(interval=16000, metric='mIoU', save_best='mIoU')
