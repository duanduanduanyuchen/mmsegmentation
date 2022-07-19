_base_ = ['./upernet_augreg_adapter_base_512x512_160k_ade20k_ss.py']
# pretrained = 'https://github.com/czczup/ViT-Adapter/releases/download/v0.1.6/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.pth'  # noqa: E501
pretrained = 'pretrained/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.pth'  # noqa: E501
model = dict(
    pretrained=pretrained,
    backbone=dict(
        type='ViTAdapter',
        img_size=384,
        pretrain_size=384,
        embed_dims=1024,
        depth=24,
        num_heads=16,
        drop_path_rate=0.4,
        deform_num_heads=16,
        with_cp=True,  # set with_cp=True to save memory
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
        window_attn=[False] * 24,
        window_size=[None] * 24),
    decode_head=dict(in_channels=[768, 768, 768, 768]),
    auxiliary_head=dict(in_channels=768))
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=2e-5,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.95))
