# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np
import torch.optim as optim

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

# orign: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
# add scaled lr parts
class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """Decay the learning rate with half-cycle cosine after warmup"""
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)  # type: ignore
        unscaled_lr =[ (1.e-8 + base_lr * lr_factor)  for base_lr in self.base_lrs]

        scaled_lr=[]
        for i,param_group in enumerate(self.optimizer.param_groups):
            if "lr_scale" in param_group:
                scaled_lr.append(unscaled_lr[i] * param_group["lr_scale"]) 
            else:
                scaled_lr.append(unscaled_lr[i])
        return scaled_lr

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= (epoch * 1.0 / self.warmup)
        return lr_factor
    