import math
import numpy as np
import torchvision.models
import torch.utils.data as data
from torchvision import transforms
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import os, torch
import torch.nn as nn
#import image_utils
import argparse, random
from functools import partial
# from opencv_flow import OpticFlow
# from resmodel import resnet18
from .CA_block import resnet18_pos_attention
# from PC_module import VisionTransformer_POS
from timm.models import create_model
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.utils.data import ConcatDataset
# from gradcam import show_cam_on_image, GradCam, preprocess_image
from torchvision.transforms import Resize
from timm.models.registry import register_model

class MMNet(nn.Module):
    def __init__(self,num_classes):
        super(MMNet, self).__init__()


        self.conv_act = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=90*2, kernel_size=3, stride=2,padding=1, bias=False,groups=1),
            nn.BatchNorm2d(180),
            nn.ReLU(inplace=True),

            )
        self.resize=Resize([14,14])
        ##main branch consisting of CA blocks
        self.main_branch =resnet18_pos_attention(num_classes=num_classes)
        num_features = self.main_branch.num_features
        self.head = nn.Sequential(
            nn.Dropout(p=0.0),
            nn.Linear(num_features, num_classes),
        )

    def forward_features(self, x):
        onset, apex=x
        # ##onset:x1 apex:x5
        # B = x1.shape[0]
        # # Position Calibration Module (subbranch)
        # POS =self.vit_pos(self.resize(x1)).transpose(1,2).view(B,512,14,14)

        act =apex -onset
        act=self.conv_act(act)
        # main branch and fusion
        out,_=self.main_branch(act,None)

        return out
    def forward(self,x ):
        x=self.forward_features(x)
        return self.head(x)
@register_model
def mmnet(pretrained=False, **kwargs):
    num_classes=kwargs.pop("num_classes", 3)
    model=MMNet(num_classes)
    return model
if __name__=='__main__':
    model = MMNet()
    x=torch.randn(1,3,224,224)
    print(model(x,x).shape)