# Copyright (c) OpenMMLab. All rights reserved.

# This script consists of several convert functions which
# can modify the weights of model in original repo to be
# pre-trained weights.

from collections import OrderedDict


def vit_converter(ckpt):
    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        if 'patch_embed' in k:
            k = k.replace('proj', 'projection')
        if 'blocks' in k:
            k = k.replace('norm1', 'ln1')
            k = k.replace('norm2', 'ln2')
            k = k.replace('mlp.fc1', 'ffn.layers.0.0')
            k = k.replace('mlp.fc2', 'ffn.layers.1')
        new_ckpt[k] = v
    return new_ckpt
