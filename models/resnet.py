import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, groups=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False,groups=groups)

## BLOCK
from .CBAM import CBAM
from .CA_block import CABlock

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # if groups != 1 or base_width != 64:
        #     raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride,groups=groups)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x1(planes, planes,groups=groups)
        self.bn2 = norm_layer(planes)
        self.attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1, stride=1,bias=False),  # 32*33*33
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.downsample = downsample
        self.stride = stride
        self.planes=planes
        self.attn=CBAM(planes)

    def forward(self, x):
        x, attn_last,if_attn =x##attn_last: downsampled attention maps from last layer as a prior knowledge
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)

        out = self.relu(out+identity)

        return out,None,True


class CBAMBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(CBAMBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # if groups != 1 or base_width != 64:
        #     raise ValueError('CBAMBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in CBAMBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride,groups=groups)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x1(planes, planes,groups=groups)
        self.bn2 = norm_layer(planes)
        self.attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1, stride=1,bias=False),  # 32*33*33
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.downsample = downsample
        self.stride = stride
        self.planes=planes
        self.attn=CBAM(planes)

    def forward(self, x):
        x, attn_last,if_attn =x##attn_last: downsampled attention maps from last layer as a prior knowledge
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)

        out = self.relu(out+identity)

        attn = self.attn(out)
        if attn_last is not None:
            attn = attn_last * attn
        # attn = attn.repeat(1, self.planes, 1, 1)
        if if_attn:
            out = out *attn

        # return out,attn[:, 0, :, :].unsqueeze(1),True
        return out,None,True


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=3, zero_init_residual=False,
                 groups=4, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        # 2023/8/15 将conv_act从融合网络中分离出来
        self.conv_act = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=90*2, kernel_size=3, stride=2,padding=1, bias=False,groups=1),
            nn.BatchNorm2d(180),
            nn.ReLU(inplace=True),
        )

        # self.inplanes = 128
        self.inplanes = 128
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(90*2, self.inplanes, kernel_size=3, stride=1,padding=1,
                               bias=False,groups=1)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2,padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, 128, layers[0],groups=1)
        self.inplanes = int(self.inplanes*1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],groups=1)
        self.inplanes = int(self.inplanes * 1)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],groups=1)
        self.inplanes = int(self.inplanes * 1)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2],groups=1)
        self.inplanes = int(self.inplanes * 1)


        self.num_features=512* block.expansion
        # self.fc = nn.Linear(512* block.expansion*196, num_classes)
        self.drop = nn.Dropout(p=0.1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck):
        #             nn.init.constant_(m.bn3.weight, 0)
        #         elif isinstance(m, CBAMBlock):
        #             nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False,groups=1):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # 2023/8/15 将conv_act从融合网络中分离出来
        x=self.conv_act(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        ## main branch
        x,attn1,_ = self.layer1((x,None,True))
        if attn1 is not None:
            attn1 = self.maxpool(attn1)

        x ,attn2,_= self.layer2((x,attn1,True))
        if attn2 is not None:
            attn2=self.maxpool(attn2)

        x ,attn3,_= self.layer3((x,attn2,True))
        if attn3 is not None:
            attn3 = self.maxpool(attn3)
        x,attn4,_ = self.layer4((x,attn3,True))

        return x

    def forward(self, x):
        return self._forward_impl(x)


if __name__=='__main__':
    model=ResNet(CBAMBlock,[3, 4, 6, 3])
    x=torch.randn(1,180,112,112)
    y=model(x)
    print(y.shape)