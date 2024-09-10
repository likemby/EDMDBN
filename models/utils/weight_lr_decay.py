# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ELECTRA https://github.com/google-research/electra
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import json
from typing import Union

def build_param_groups(model, weight_decay=1e-5,use_lr_scale=False,prefix=''):
    """Parameter groups for layer-wise lr decay and weight decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58 
     and timm optim_factory.add_weight_decay
    """
    param_group_names={}
    param_groups={}
    no_decay_names=set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue # frozen weights

        # no decay: all 1D parameters and model specific ones
        if len(param.shape) == 1 or name.endswith(".bias")  or  hasattr(param,'no_weight_decay'):
            g_decay = "no_decay"
            this_decay = 0.
            no_decay_names.add(name)
        else:
            g_decay = "decay"
            this_decay = weight_decay
        

        # lr scale: accoording the model parameter attribute "lr_scale", maybe not available 
        lr_scale=param.lr_scale if (hasattr(param,'lr_scale') and use_lr_scale)  else 1.0
        group_name = "{}&lr_scale:{}".format(g_decay,lr_scale) 
        group_name = "-".join([prefix,group_name])

        if group_name not in param_group_names:
            this_scale = lr_scale

            param_group_names[group_name] = {
                "name":group_name ,
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "name":group_name, # for pytorch lighting lr monitor visulization
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
        
        param_group_names[group_name]["params"].append(name)
        param_groups[group_name]["params"].append(param)
    
    if not use_lr_scale:
        [ x.pop('lr_scale') for x in param_groups.values() ]

    # print('no decay name list: {}'.format(no_decay_names))
    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())

def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    else:
        return num_layers