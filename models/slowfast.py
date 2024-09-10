from timm.models.registry import register_model
from timm.models.layers import DropPath, trunc_normal_
import torch
from pytorchvideo.models.slowfast import create_slowfast


@register_model
def slowfast18(pretrained=False, **kwargs):
    num_classes=kwargs.pop("num_classes", 3)
    model=create_slowfast(model_num_class=num_classes,model_depth=50,head_pool_kernel_sizes= ((2, 7, 7), (8, 7, 7)))
    return model
if __name__ == '__main__':
    model=slowfast18()
    slow=torch.randn(2,3,2,224,224)
    fast=torch.randn(2,3,8,224,224)
    res=model([slow,fast])
    print()

