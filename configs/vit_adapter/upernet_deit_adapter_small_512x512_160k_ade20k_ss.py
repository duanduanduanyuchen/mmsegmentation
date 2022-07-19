_base_ = ['./upernet_deit_adapter_tiny_512x512_160k_ade20k_ss.py']
# pretrained = 'https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth'  # noqa: E501
pretrained = 'pretrained/deit_small_patch16_224-cd65a155.pth'
model = dict(
    pretrained=pretrained,
    backbone=dict(
        type='ViTAdapter', embed_dims=384, num_heads=6, drop_path_rate=0.2),
    decode_head=dict(in_channels=[384, 384, 384, 384]),
    auxiliary_head=dict(in_channels=384))
optimizer = dict(type='AdamW', lr=6e-5)
