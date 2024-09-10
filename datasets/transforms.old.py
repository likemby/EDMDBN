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

class CASME2AugTransform(nn.Module):
    """
    [Image ... ] => T,C,H,W => C,T,H,W
    (is_train,input_size=224,num_frames=16,*,interpolation_strategy=FixSequenceInterpolation)
    """
    def __init__(self,is_train,input_size=224,num_frames=16,*,interpolation_strategy='FixSequenceInterpolation',use_optical_flow=False,use_RGB=True) -> None:
        super().__init__()
        self.head_transform=[ImglistToTensor()]
        train_transform=[
            T.Resize(input_size,InterpolationMode.BICUBIC),
            T.RandomResizedCrop(input_size, scale=(0.8,1.0),ratio=(0.9,1.1),interpolation=InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
        ]
        val_transform=[
            T.Resize(input_size,InterpolationMode.BICUBIC),
            T.CenterCrop(input_size)# type: ignore
        ]
        self.space_transforms= train_transform if is_train else val_transform
        self.time_transforms=[
            TimeInterpolation(num_frames,interpolation_strategy)
            ]

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

        return T.Compose(self.head_transform + self.space_transforms + self.time_transforms+ self.tail_transforms)(x)


class MERRGBTransform(nn.Module):
    """
    [Image ... ] => T,C,H,W => C,T,H,W
    (is_train,input_size=(224,224),num_frames=16,*,interpolation_strategy='FixSequenceInterpolation)
    """
    def __init__(self,is_train,input_size=224,num_frames=16,*,interpolation_strategy='FixSequenceInterpolation',use_optical_flow=False,use_RGB=True) -> None:
        super().__init__()
        self.head_transform=[ImglistToTensor()]
        input_size=(input_size,input_size)
        train_transform=[
            T.Resize(input_size,InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            # T.RandomResizedCrop(input_size, scale=(0.8,1.0),ratio=(0.9,1.1),interpolation=InterpolationMode.BICUBIC),
            T.RandomRotation(4),
            T.RandomCrop(input_size, padding=4,padding_mode='edge')
        ]
        
        val_transform=[
            T.Resize(input_size,InterpolationMode.BICUBIC)
        ]
        self.space_transforms= train_transform if is_train else val_transform
        self.time_transforms=[
            TimeInterpolation(num_frames,interpolation_strategy)
            ]

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
        res=T.Compose(self.head_transform + self.space_transforms + self.time_transforms+ self.tail_transforms)(x)
        return res
class CASME2CleanTransform(MERRGBTransform):
    def __init__(self,is_train,input_size=224,num_frames=16,*,interpolation_strategy='FixSequenceInterpolation',use_optical_flow=False,use_RGB=True) -> None:
        super().__init__(is_train,input_size,num_frames,interpolation_strategy=interpolation_strategy,use_optical_flow=use_optical_flow,use_RGB=use_RGB)
        input_size=(input_size,input_size)
        
        train_transform=[
            T.Resize(input_size,InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomResizedCrop(input_size, scale=(0.8,1.0),ratio=(0.9,1.1),interpolation=InterpolationMode.BICUBIC),
            # T.RandomRotation(4),
            # T.RandomCrop(input_size, padding=4,padding_mode='edge'),

        ]
        val_transform=[
            T.Resize(input_size,InterpolationMode.BICUBIC)
        ]
        self.space_transforms= train_transform if is_train else val_transform

class SeprateOnsetApex(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self,x:torch.Tensor,*,vis=False):
        # x: CTHW
        return x[:,0,...],x[:,1,...]

class PrepareFuseData(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self,x:torch.Tensor,*,vis=False):
        # x: CTHW，from onset to apex  
        apex_sub_onset=x[:,-1,...] - x[:,0,...] 
        framesdiff_or_rgbimgs=x[:,1:,...]-x[:,:-1,...]
        return apex_sub_onset,framesdiff_or_rgbimgs
    
class SAMMTransform(MERRGBTransform):
    def __init__(self,is_train,input_size=224,num_frames=16,*,interpolation_strategy='FixSequenceInterpolation',use_optical_flow=False,use_RGB=True) -> None:
        super().__init__(is_train,input_size,num_frames,interpolation_strategy=interpolation_strategy,use_optical_flow=use_optical_flow,use_RGB=use_RGB)
        
        train_transform=[
            T.Resize(input_size,InterpolationMode.BICUBIC),
            T.RandomResizedCrop(input_size, scale=(0.9,1.0),ratio=(0.9,1.1),interpolation=InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(4),
            T.RandomCrop(input_size, padding=4,padding_mode='edge'),
        ]
        input_size=(input_size,input_size)
        
        val_transform=[
            T.Resize(input_size,InterpolationMode.BICUBIC)
        ]
        self.space_transforms= train_transform if is_train else val_transform
        # assert False
        self.tail_transforms.append(SeprateOnsetApex())

class SAMMSpaceTransform(MERRGBTransform):
    def __init__(self,is_train,input_size=224,num_frames=16,*,interpolation_strategy='FixSequenceInterpolation',use_optical_flow=False,use_RGB=True) -> None:
        super().__init__(is_train,input_size,num_frames,interpolation_strategy=interpolation_strategy,use_optical_flow=use_optical_flow,use_RGB=use_RGB)
        
        train_transform=[
            T.Resize(input_size,InterpolationMode.BICUBIC),
            T.RandomResizedCrop(input_size, scale=(0.9,1.0),ratio=(0.9,1.1),interpolation=InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(4),
            T.RandomCrop(input_size, padding=4,padding_mode='edge'),
        ]
        input_size=(input_size,input_size)
        
        val_transform=[
            T.Resize(input_size,InterpolationMode.BICUBIC)
        ]
        self.space_transforms= train_transform if is_train else val_transform
        self.time_transforms=[]
        # assert False
        self.tail_transforms.append(SeprateOnsetApex())

class FuseNetTransform(MERRGBTransform):
    def __init__(self,is_train,input_size=224,num_frames=16,*,interpolation_strategy='FixSequenceInterpolation',use_optical_flow=False,use_RGB=True) -> None:
        super().__init__(is_train,input_size,num_frames,interpolation_strategy=interpolation_strategy,use_optical_flow=use_optical_flow,use_RGB=use_RGB)
        input_size=(input_size,input_size)
        
        train_transform=[
            T.Resize(input_size,InterpolationMode.BICUBIC),
            # T.RandomResizedCrop(input_size, scale=(0.9,1.0),ratio=(0.9,1.1),interpolation=InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(4),
            T.RandomCrop(input_size, padding=4,padding_mode='edge'),
        ]
        
        val_transform=[
            T.Resize(input_size,InterpolationMode.BICUBIC)
        ]
        self.space_transforms= train_transform if is_train else val_transform
        self.time_transforms=[
            TimeInterpolation(num_frames+1,interpolation_strategy)
        ]
        self.tail_transforms.append(PrepareFuseData())
# class FuseNetTransform(MERRGBTransform):
#     def __init__(self,is_train,input_size=224,num_frames=16,*,interpolation_strategy='FixSequenceInterpolation',use_optical_flow=False,use_RGB=True) -> None:
#         super().__init__(is_train,input_size,num_frames,interpolation_strategy=interpolation_strategy,use_optical_flow=use_optical_flow,use_RGB=use_RGB)
        
#         train_transform=[
#             T.Resize(input_size,InterpolationMode.BICUBIC),
#             T.RandomResizedCrop(input_size, scale=(0.9,1.0),ratio=(0.9,1.1),interpolation=InterpolationMode.BICUBIC),
#             T.RandomHorizontalFlip(p=0.5),
#             T.RandomRotation(4),
#             T.RandomCrop(input_size, padding=4,padding_mode='edge'),
#         ]
#         input_size=(input_size,input_size)
        
#         val_transform=[
#             T.Resize(input_size,InterpolationMode.BICUBIC)
#         ]
#         self.space_transforms= train_transform if is_train else val_transform
#         self.time_transforms=[
#             TimeInterpolation(num_frames+1,interpolation_strategy)
#         ]
#         self.tail_transforms.append(PrepareFuseData())


class CASME2OpticalTransform(nn.Module):
    def __init__(self,is_train,input_size=(224,224),num_frames=16,*,interpolation_strategy='FixSequenceInterpolation',use_optical_flow=False,use_RGB=True) -> None:
        super().__init__()
        self.head_transform=[ImglistToTensor(T.PILToTensor)]
        train_transform=[
            T.Resize(input_size,InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(4),
            T.RandomCrop(input_size, padding=4)
        ]
        val_transform=[
            T.Resize(input_size,InterpolationMode.BICUBIC)
        ]
        self.space_transforms= train_transform if is_train else val_transform
        self.time_transforms=[
            # TimeInterpolation(num_frames,interpolation_strategy)
            ]
        self.optical_flow_transform=[OpticalFlow()]
        self.tail_transforms=[
            T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ConvertBCHWtoCBHW()
        ]
    def forward(self,x:typing.List[Image.Image],*,vis=False):
        """
        pass
        """
        if vis:
            from torchvision.utils import flow_to_image
            t=T.Compose(self.head_transform + self.space_transforms+ self.time_transforms+self.optical_flow_transform)
            imgs=t(x)
            return [ T.ToPILImage()(flow_to_image(img)) for img in imgs ]

        return T.Compose(self.head_transform + self.space_transforms + self.time_transforms+self.optical_flow_transform+ self.tail_transforms)(x)

from merlib.data.base import CASME2_info,SAMM_info,SMICHSE_info
def get_me_transform(is_train,args):
    assert args.transform_name, 'transform name is not set'
    input_size,  num_frames= getattr(args,'input_size') , getattr(args,'num_frames')

    return FuseNetTransform(is_train,input_size=input_size,num_frames=num_frames)
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

# def get_me_transform(input_size, train, num_samples, optical_flow, sample_strategy, scale,registration_method):
#     """
#     用于有监督时的微表情训练，测试
#     """
#     if input_size <= 224:
#         crop_pct = 224 / 256
#     else:
#         crop_pct = 1.0
#     size = int(input_size / crop_pct)
#     # input_size=(input_size,input_size)
#     me_transform = [ImglistToTensor()]
#     # size=(256,256)
#     if train:
#         me_transform.extend([
#             T.Resize(size, interpolation=InterpolationMode.BICUBIC),  # type: ignore
#             T.CenterCrop(input_size),
#             T.RandomHorizontalFlip(p=0.5),
#             T.RandomResizedCrop(input_size, scale=scale,ratio=(0.9,1.1),interpolation=InterpolationMode.BICUBIC),
#             # T.RandomRotation(5,interpolation=InterpolationMode.BICUBIC),
        
#         ])  # type: ignore
#     else:
#         me_transform.extend([
#             T.Resize(size, interpolation=InterpolationMode.BICUBIC),  # type: ignore
#             T.CenterCrop(input_size)# type: ignore
#         ])  

#     num_samples += 1 if optical_flow else 0
#     if sample_strategy == 'UniformTemporalSubsample':
#         me_transform.append(UniformTemporalSubsample(
#             num_samples, temporal_dim=-4))  # type: ignore
#     elif sample_strategy == 'FixSequenceInterpolation':
#         me_transform.append(FixSequenceInterpolation(num_samples))  # type: ignore
#     else:
#         assert False

#     if optical_flow:
#         me_transform.append(
#             OpticalFlow()  # type: ignore
#         )
#     else:
#         me_transform.append(
#             T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)  # type: ignore
#         )

#     me_transform.append(ConvertBCHWtoCBHW())  # type: ignore
#     return T.Compose(me_transform)

if __name__=='__main__':
    CASME2CleanTransform(True,224,16)([Image.open('/home/mby/computer_vision/database/SAMM/dlib_crop_twice/006/006_1_2/5562_5632/5562.jpg')])