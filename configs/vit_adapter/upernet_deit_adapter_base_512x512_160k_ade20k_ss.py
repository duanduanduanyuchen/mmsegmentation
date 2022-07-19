_base_ = ['./upernet_deit_adapter_tiny_512x512_160k_ade20k_ss.py']
# pretrained = 'https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth'  # noqa: E501
pretrained = 'pretrained/deit_base_patch16_224-b5f2ef4d.pth'
model = dict(
    pretrained=pretrained,
    backbone=dict(
        type='ViTAdapter',
        embed_dims=768,
        num_heads=12,
        drop_path_rate=0.3,
        deform_num_heads=12,
        deform_ratio=0.5,
    ),
    decode_head=dict(in_channels=[768, 768, 768, 768]),
    auxiliary_head=dict(in_channels=768))
optimizer = dict(type='AdamW', lr=6e-5)
