import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import nn
import torch
from merlib.datasets.transforms import ConvertBCHWtoCBHW,ImglistToTensor,TimeInterpolation,OpticalFlow
import typing 
from PIL import Image

from pytorchvideo.transforms import (
    RandomShortSideScale,
    Div255,
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)
class PrepareFuseData(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self,x:torch.Tensor,*,vis=False):
        # x: CTHW，from onset to apex  
        apex_sub_onset=x[:,-1,...] - x[:,0,...] 
        framesdiff_or_rgbimgs=x[:,1:,...]-x[:,:-1,...]
        return apex_sub_onset,framesdiff_or_rgbimgs
    
class FuseNetTransform(nn.Module):
    """
    [Image ... ] => T,C,H,W => C,T,H,W
    (is_train,input_size=(224,224),num_frames=16,*,interpolation_strategy='FixSequenceInterpolation)
    """
    def __init__(self,is_train,input_size=224,num_frames=16,*,interpolation_strategy='FixSequenceInterpolation',frames_mode=0,rgb_diff_abs=False) -> None:
        super().__init__()
        self.frames_mode=frames_mode
        self.rgb_diff_abs=rgb_diff_abs
        assert self.frames_mode in [0,1,2]
        self.head_transform=[ImglistToTensor()]
        input_size=(input_size,input_size)
        train_transform=[
            T.Resize(input_size,InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            # T.RandomResizedCrop(input_size, scale=(0.8,1.0),ratio=(0.9,1.1),interpolation=InterpolationMode.BICUBIC),#启用，测试五分类效果
            T.RandomRotation(4),
            T.RandomCrop(input_size, padding=4,padding_mode='edge')
        ]
        
        val_transform=[
            T.Resize(input_size,InterpolationMode.BICUBIC)
        ]
        self.space_transforms= train_transform if is_train else val_transform
        if self.frames_mode in [0,1]:
            self.time_transforms=[
                TimeInterpolation(num_frames+1,interpolation_strategy)
            ]
        elif self.frames_mode==2:
            self.time_transforms=[
                TimeInterpolation(num_frames,interpolation_strategy),
            ]
        else:
            assert False

        self.tail_transforms=[
            T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ConvertBCHWtoCBHW(),
        ]

    def forward(self,x:typing.List[Image.Image],*,vis=False):
        """
        pass
        """
        if vis:
            t=T.Compose(self.head_transform + self.space_transforms+ self.time_transforms)
            imgs=t(x)
            return [ T.ToPILImage()(img) for img in imgs ]
        x=T.Compose(self.head_transform + self.space_transforms)(x)
        onset_and_apex=T.Compose(self.tail_transforms)(x[:2,...]) # CTHW，假设x的前两帧是onset和apex
        frames=T.Compose(self.time_transforms+ self.tail_transforms)(x) # CTHW，假设x是from onset to apex
        
        apex_sub_onset=onset_and_apex[:,-1,...] - onset_and_apex[:,0,...]
        if self.rgb_diff_abs:
            assert not self.rgb_diff_abs 
            apex_sub_onset=torch.abs(apex_sub_onset)
            pass
        if self.frames_mode==0:
            framesdiff_or_rgbimgs=frames[:,1:,...]-frames[:,:-1,...]
        elif self.frames_mode==1:       
            framesdiff_or_rgbimgs=frames[:,1:,...]-frames[:,0:1,...]
        elif self.frames_mode==2:
            framesdiff_or_rgbimgs=frames

        return_tensor= torch.cat((apex_sub_onset.unsqueeze(1),framesdiff_or_rgbimgs,onset_and_apex[:,1,...].unsqueeze(1)),dim=1)
        return return_tensor

class FuseNetTransform2(nn.Module):
    """
    为了可视化暂用
    [Image ... ] => T,C,H,W => C,T,H,W
    (is_train,input_size=(224,224),num_frames=16,*,interpolation_strategy='FixSequenceInterpolation)
    """
    def __init__(self,is_train,input_size=224,num_frames=16,*,interpolation_strategy='FixSequenceInterpolation',frames_mode=0,rgb_diff_abs=False) -> None:
        super().__init__()
        self.frames_mode=frames_mode
        self.rgb_diff_abs=rgb_diff_abs
        assert self.frames_mode in [0,1,2]
        self.head_transform=[ImglistToTensor()]
        input_size=(input_size,input_size)
        train_transform=[
            T.Resize(input_size,InterpolationMode.BICUBIC),
            # T.RandomHorizontalFlip(p=0.5),
            # T.RandomResizedCrop(input_size, scale=(0.8,1.0),ratio=(0.9,1.1),interpolation=InterpolationMode.BICUBIC),
            T.RandomRotation(4),
            T.RandomCrop(input_size, padding=4,padding_mode='edge')
        ]
        
        val_transform=[
            T.Resize(input_size,InterpolationMode.BICUBIC)
        ]
        self.space_transforms= train_transform if is_train else val_transform
        if self.frames_mode in [0,1]:
            self.time_transforms=[
                TimeInterpolation(num_frames+1,interpolation_strategy)
            ]
        elif self.frames_mode==2:
            self.time_transforms=[
                TimeInterpolation(num_frames,interpolation_strategy),
            ]
        else:
            assert False

        self.tail_transforms=[
            T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ConvertBCHWtoCBHW(),
        ]

    def forward(self,x:typing.List[Image.Image],*,vis=False):
        """
        pass
        """
        if vis:
            t=T.Compose(self.head_transform + self.space_transforms+ self.time_transforms)
            imgs=t(x)
            return [ T.ToPILImage()(img) for img in imgs ]
        x=T.Compose(self.head_transform + self.space_transforms)(x)
        onset_and_apex=T.Compose(self.tail_transforms)(x[:2,...]) # CTHW，假设x的前两帧是onset和apex
        frames=T.Compose(self.time_transforms+ self.tail_transforms)(x) # CTHW，假设x是from onset to apex
        
        apex_sub_onset=onset_and_apex[:,-1,...] - onset_and_apex[:,0,...]
        if self.rgb_diff_abs:
            assert not self.rgb_diff_abs 
            apex_sub_onset=torch.abs(apex_sub_onset)
            pass
        if self.frames_mode==0:
            framesdiff_or_rgbimgs=frames[:,1:,...]-frames[:,:-1,...]
        elif self.frames_mode==1:       
            framesdiff_or_rgbimgs=frames[:,1:,...]-frames[:,0:1,...]
        elif self.frames_mode==2:
            framesdiff_or_rgbimgs=frames

        return_tensor= torch.cat((apex_sub_onset.unsqueeze(1),framesdiff_or_rgbimgs,onset_and_apex[:,1,...].unsqueeze(1)),dim=1)
        return return_tensor


from merlib.data.base import CASME2_info,SAMM_info,SMICHSE_info
def get_me_transform(is_train,args):
    assert args.transform_name, 'transform name is not set'
    input_size,  num_frames= getattr(args,'input_size') , getattr(args,'num_frames')
    frames_mode=getattr(args,'frames_mode',0)
    rgb_diff_abs=getattr(args,'rgb_diff_abs',False)
    # 控制帧采样方式和插值策略
    # 0-all frames, 1-apex_frame, 2-onset,apex frames, 3-onset,apex,offsetframes, 4-frames_perseqment
    assert args.sampling_strategy in [2, 3]
    return FuseNetTransform(is_train,input_size=input_size,num_frames=num_frames,frames_mode=frames_mode,rgb_diff_abs=rgb_diff_abs)
    if args.dataset_name==CASME2_info.name:
        return CASME2CleanTransform(is_train,input_size=input_size,num_frames=num_frames)
    if args.dataset_name==SAMM_info.name:
        # print('use SAMM transform')
        # return SAMMTransform(is_train,input_size=input_size,num_frames=num_frames)
        print('use SAMMSpaceTransform transform')
        return SAMMSpaceTransform(is_train,input_size=input_size,num_frames=num_frames)
    if args.dataset_name==SMICHSE_info.name:
        # print('use SAMM transform')
        # return SAMMTransform(is_train,input_size=input_size,num_frames=num_frames)
        print('use SAMMSpaceTransform transform')
        return SAMMSpaceTransform(is_train,input_size=input_size,num_frames=num_frames)
    assert False
