    
import torch.nn as nn
import torch
from pathlib import Path
from merlib.helper import get_print_logger
from timm.models.registry import register_model
from timm.models import create_model
from torchvision.transforms import Resize
from functools import partial

if __name__=='__main__':
    from swin_transformer_3d2 import SwinTransformer3D
    from resnet import BasicBlock,ResNet, CBAMBlock,CABlock
    from PC_module import VisionTransformer_POS
else:
    from .swin_transformer_3d2 import SwinTransformer3D
    from .resnet import BasicBlock,ResNet, CBAMBlock,CABlock,conv1x1
    from .PC_module import VisionTransformer_POS

class AddModule(nn.Module):
    def __init__(self):
        super(AddModule, self).__init__()

    def forward(self, a, b):
        return a + b

def custom_swin(pretrained=False, **kwargs):
    if pretrained:
        pretrained="~/checkpoints/swin_tiny_patch4_window7_224_22k.pth"
        pretrained=Path(pretrained).expanduser().as_posix()
    drop_path=kwargs.pop("drop_path", 0.2)
    window_size=kwargs.pop('window_size',(8, 7, 7))
    patch_size=kwargs.pop('patch_size',(8, 4, 4))
    in_chans=kwargs.pop('in_chans',3)
    # in_chans=kwargs.pop('in_chans',96)
    # embed_dim=kwargs.pop('embed_dim',96)
    embed_dim=kwargs.pop('embed_dim',128)
    depth=kwargs.pop('depth',4)
    depths=[2, 2, 6, 2]
    # num_heads=[3, 6, 12, 24]
    num_heads=[4, 8, 16, 32]
    use_atten=kwargs.pop('use_atten',False)
    print("use_atten:",use_atten)
    model = SwinTransformer3D(patch_size=patch_size,  # (2, 4, 4)
                              embed_dim=embed_dim,
                              in_chans=in_chans,
                              depths=depths[:depth],
                              num_heads=num_heads[:depth],
                              window_size=window_size,
                              mlp_ratio=4.0,
                              qkv_bias=True,
                              qk_scale=None,
                              drop_rate=0.0,
                              attn_drop_rate=0.0,
                              drop_path_rate=drop_path,
                              patch_norm=True,
                              pretrained=pretrained,
                              use_atten=use_atten,
                                **kwargs)
    return model

def custom_resnet(pretrained=False, progress=True, **kwargs):
    block_name=kwargs.pop('block_name','BasicBlock')
    if block_name=='BasicBlock':
        block=BasicBlock
    elif block_name=='CBAMBlock':
        block=CBAMBlock
    elif block_name=='CABlock':
        block=CABlock
    else:
        raise NotImplementedError
    layers=[1, 1, 1, 1]
    model = ResNet(block, layers, **kwargs)
    return model

class LongShortActionFuseNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.num_classes=kwargs.pop('num_classes',3)
        self.use_long_action=kwargs.pop('use_long_action',False)
        self.use_short_action=kwargs.pop('use_short_action',False)
        self.use_pos_vit=kwargs.pop('use_pos_vit',False)
        assert self.use_long_action or self.use_short_action,"use_long_action and use_short_action can't be both False"
        self.concat=kwargs.pop('use_feature_concat',False) # if False, then use element-wise add
        self.use_feature_conv=kwargs.pop('use_feature_conv',False)
        self.use_gap=kwargs.pop('use_gap',False)
        self.use_gmp=kwargs.pop('use_gmp',False)
        self.use_atten=kwargs.pop('use_swin_CA',False)

        # 移动到具体的模型中
        # self.conv_act = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=90*2, kernel_size=3, stride=2,padding=1, bias=False,groups=1),
        #     nn.BatchNorm2d(180),
        #     nn.ReLU(inplace=True),
        # )
        # self.slow_act = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1,padding=1, bias=False,groups=1),
        #     nn.BatchNorm2d(96),
        #     nn.ReLU(inplace=True),
        # )

        self.vit_pos=VisionTransformer_POS(img_size=112,
            patch_size=8, embed_dim=512, depth=2, num_heads=4, mlp_ratio=4, qkv_bias=True,norm_layer=partial(nn.LayerNorm, eps=1e-6),drop_path_rate=0.)
        self.resize=Resize([112,112])

        # 2023-8-14: self.long_action_model=custom_resnet(pretrained=False, num_classes=self.num_classes) if self.use_long_action else None 
        self.long_action_model=custom_swin(pretrained=False,depth=3,use_atten=self.use_atten) if self.use_long_action else None

        self.short_action_model=custom_swin(pretrained=False,depth=3,use_atten=self.use_atten) if self.use_short_action else None

        self.add_operator=AddModule()
        num_features=None
        if self.use_long_action and self.use_short_action:
            short_action_net_num_features = self.short_action_model.num_features
            long_action_net_num_features = self.long_action_model.num_features
            if self.concat:
                print("use feature concat")
                num_features=short_action_net_num_features + long_action_net_num_features
                self.feature_conv=nn.Sequential(
                    conv1x1(num_features,num_features),
                    nn.BatchNorm2d(num_features),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.2)
                )  if self.use_feature_conv else nn.Identity()
            else:
                assert short_action_net_num_features==long_action_net_num_features,"short_action_net_num_features: {}, long_action_net_num_features: {}".format(short_action_net_num_features,long_action_net_num_features)
                num_features=long_action_net_num_features
        elif self.use_long_action:
            num_features=self.long_action_model.num_features
        elif self.use_short_action:
            num_features=self.short_action_model.num_features
        else:
            raise Exception("use_long_action and use_short_action can't be both False")
        
        assert num_features is not None
        if self.use_gap:
            print("use global average pooling")
            self.to_feature=nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                          nn.Flatten())
            self.head = nn.Sequential(
                nn.Dropout(p=0.0),
                nn.Linear(num_features, self.num_classes)
            )
        elif self.use_gmp:
            print("use global max pooling")
            self.to_feature=nn.Sequential(nn.AdaptiveMaxPool2d((1,1)),
                                          nn.Flatten())
            self.head = nn.Sequential(
                nn.Dropout(p=0.0),
                nn.Linear(num_features, self.num_classes)
            )
        else:
            self.to_feature=nn.Flatten()
            self.head = nn.Sequential(
            nn.Dropout(p=0.0),
            nn.Linear(num_features*14*14, self.num_classes)
        )

    def forward_features_with_shape(self, x):
        # print([i.shape for i in x])
        apex_sub_onset, framesdiff_or_rgbimgs,onset_frame=x[:,:,0,...],x[:,:,1:-1,...],x[:,:,-1,...]
        
        if self.use_pos_vit:
            B = onset_frame.shape[0]
            POS =self.vit_pos(self.resize(onset_frame)).transpose(1,2).view(B,512,14,14)

        # long_action
        long_action=None
        if self.use_long_action:
            # 2023-8-14: long_action=self.conv_act(apex_sub_onset)
            long_action=apex_sub_onset.unsqueeze(2)
            # print("20023/8/15 long_action:",long_action.shape)
            long_action=self.long_action_model(long_action) # B,C,T,H,W
            if len(long_action.shape)==5:
                long_action=long_action.mean(dim=2)
            if self.use_pos_vit:
                long_action=long_action+POS

        # short_action
        if self.use_short_action:
            # B,C, T, H, W = framesdiff_or_rgbimgs.shape
            # 将输入数据重塑为 (B*T, C, H, W)
            short_action=self.short_action_model(framesdiff_or_rgbimgs) # B,C,T,H,W
            short_action=short_action.mean(dim=2) # B,C,H,W
            if self.use_pos_vit:
                short_action=short_action+POS
        
        # fuse
        if self.use_long_action and self.use_short_action:
            if self.concat:
                out=torch.cat((long_action,short_action),dim=1)
                if self.feature_conv:
                    out=self.feature_conv(out)
            else:
                out=self.add_operator(long_action,short_action)
                # out=long_action+short_action
        elif self.use_long_action:
            out=long_action
        elif self.use_short_action:
            out=short_action
        else:   
            raise Exception("use_long_action and use_short_action can't be both False")

        return out
    def forward_features(self, x):
        x=self.forward_features_with_shape(x)
        x=self.to_feature(x)
        return x
    
    def forward(self,x ):
        x=self.forward_features(x)
        x=self.head(x)
        return x
    
@register_model
def fusenet(pretrained=False, **kwargs):
    model = LongShortActionFuseNet(**kwargs)
    return model

if __name__== '__main__':
    model=create_model('fusenet',num_classes=3, use_long_action=True,use_short_action=False,concat=False)
    apex_sub_onset=torch.randn(2,3,224,224)
    framesdiff_or_rgbimgs=torch.randn(2,3,8,224,224)
    apex_frame=torch.randn(2,3,224,224)
    out=model((apex_sub_onset,framesdiff_or_rgbimgs,apex_frame))
    print(out.shape)
