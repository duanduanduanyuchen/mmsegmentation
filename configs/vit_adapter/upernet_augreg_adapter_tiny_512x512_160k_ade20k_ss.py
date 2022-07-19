_base_ = ['./upernet_deit_adapter_tiny_512x512_160k_ade20k_ss.py']
# pretrained = 'https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.pth'  # noqa: E501
pretrained = 'pretrained/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.pth'  # noqa: E501
model = dict(pretrained=pretrained)
