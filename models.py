"""
Code inspired by:
https://pytorch.org/vision/stable/_modules/torchvision/models/mobilenetv2.html
https://pytorch.org/vision/stable/_modules/torchvision/models/mobilenetv3.html
"""
import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Sequence,Tuple
from torch import nn, Tensor
import warnings

class CausalModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = None

    def reset_activation(self) -> None:
        self.activation = None


class TemporalCGAvgPool3D(CausalModule):
    def __init__(self,):
        super().__init__()
        self.T = 0 
        self.register_forward_hook(self._detach_activation)

    def forward(self, x: Tensor):
        self.T += x.shape[2]
        if self.activation is None:
            self.activation = torch.sum(x,dim =2)
        else:
            self.activation += torch.sum(x,dim =2)

        x = self.activation/self.T
        return x
    @staticmethod
    def _detach_activation(module : nn.Module,input: Tensor,output: Tensor) -> None:
        module.activation.detach_()


    def reset_activation(self) -> None:
        super().reset_activation()  
        self.T = 0


class CausalConv(CausalModule):
    def __init__(
            self,
            in_planes : int,
            out_planes : int,
            kernel_size : Tuple,
            **kwargs : Any,
            ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.dim_pad = self.kernel_size[0]-1
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size, **kwargs)
    def forward(self, x : Tensor) -> Tensor:
        device = x.device
        if self.dim_pad <= 0:
            x = self.conv(x)
        else:
            if self.activation is None:
                self._setup_activation(x.shape)
            x = torch.cat((self.activation.to(device),x),2)
            self._save_in_activation(x)
            x = self.conv(x)
        return x
    def _save_in_activation(self, x : Tensor) -> Tensor:
        assert self.dim_pad > 0
        self.activation = x[:,:,-self.dim_pad:,...].clone().detach().cpu()
    def _setup_activation(self, input_shape : Tuple) -> None:
        assert self.dim_pad > 0
        self.activation = torch.zeros(*input_shape[:2],self.dim_pad,*input_shape[3:]  )
        



class SqueezeExcitation(nn.Module):

    def __init__(self, input_channels: int, squeeze_factor: int = 4):
        super().__init__()
        squeeze_channels = _make_divisible(input_channels//squeeze_factor,8)
        self.fc1 = CausalConv(input_channels, squeeze_channels, (1,1,1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = CausalConv(squeeze_channels, input_channels, (1,1,1))

    def _scale(self, input: Tensor, inplace: bool) -> Tensor:
        scale = F.adaptive_avg_pool3d(input, 1)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        return F.hardsigmoid(scale, inplace=inplace)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input, True)
        return scale * input

def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        *,
        kernel_size: Tuple,
        padding : Tuple,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if activation_layer is None:
            activation_layer = nn.Hardswish
        super(ConvBNActivation, self).__init__(
            CausalConv(in_planes, out_planes, 
                      kernel_size = kernel_size, 
                      stride = stride, 
                      padding = padding, 
                      groups=groups,
                      bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )
        self.out_channels = out_planes
        
        


class BasicBneck(nn.Module):
    def __init__(self, cfg : "CfgNode") -> None:
        super().__init__()
        assert type(cfg.stride) is tuple
        layers_first = []
        layers = []
        self.block_first = None
        if not (cfg.stride == (1,1,1) and cfg.input_channels == cfg.out_channels):
            """From the paper is not clar to how this part is contructed.
                "We also apply skip connections that are traditionally used in ResNets, adding a 1x1x1 convolution in the first 
                layer of each block which may change the base channels or downsample the input. However, we modify this to be 
                similar to ResNet-D where we apply 1x3x3 spatial average pooling before the convolution to improve feature 
                reppresentation"
            """ 
            layers_first.append(nn.AvgPool3d((1,3,3), stride = cfg.stride,padding = cfg.padding_avg))
            layers_first.append(ConvBNActivation(
                    in_planes = cfg.input_channels,
                    out_planes = cfg.out_channels,
                    kernel_size = (1,1,1),
                    padding = (0,0,0),
                    norm_layer= cfg.norm_layer,
                    activation_layer = cfg.activation_layer
                    ))
            self.block_first = nn.Sequential(*layers_first)

        if cfg.expanded_channels != cfg.out_channels:
            #expand
            layers.append(ConvBNActivation(
                in_planes = cfg.out_channels,
                out_planes = cfg.expanded_channels,
                kernel_size = (1,1,1),
                padding = (0,0,0),
                norm_layer= cfg.norm_layer,
                activation_layer = cfg.activation_layer
                ))
        #deepwise 
        layers.append(ConvBNActivation(
            in_planes = cfg.expanded_channels,
            out_planes = cfg.expanded_channels,
            kernel_size = cfg.kernel_size,
            padding = cfg.padding,
            stride= (1,1,1),
            norm_layer= cfg.norm_layer,
            groups = cfg.expanded_channels,
            activation_layer = cfg.activation_layer
            ))
        #SE
        layers.append(SqueezeExcitation(cfg.expanded_channels))
        #project
        layers.append(ConvBNActivation(
            cfg.expanded_channels,
            cfg.out_channels,
            kernel_size = (1,1,1),
            padding = (0,0,0),
            norm_layer = cfg.norm_layer,
            activation_layer = nn.Identity
            ))
        self.block = nn.Sequential(*layers)
        #ReZero
        self.alpha = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.out_channels = cfg.out_channels
    def forward(self, input: Tensor) -> Tensor:
        if self.block_first is not None:
            input = self.block_first(input)
        result = self.block(input)
        result = input + self.alpha *result
        return result

class MoViNet(nn.Module):
    def __init__(self, 
            cfg : "CfgNode",
            num_classes : int) -> None:
        super().__init__()
        layers = []
        #conv1
        self.conv1 = ConvBNActivation(
            in_planes = cfg.conv1.input_channels,
            out_planes = cfg.conv1.out_channels,
            kernel_size = cfg.conv1.kernel_size,
            stride = cfg.conv1.stride,
            norm_layer= cfg.conv1.norm_layer,
            padding = cfg.conv1.padding,
            activation_layer = cfg.conv1.activation_layer
            )
        #blocks
        for block in cfg.blocks:
            for basicblock in block:
                layers.append(BasicBneck(basicblock))
        self.blocks = nn.Sequential(*layers)
        #conv7
        self.conv7 = ConvBNActivation(
            in_planes = cfg.conv7.input_channels,
            out_planes = cfg.conv7.out_channels,
            kernel_size = cfg.conv7.kernel_size,
            stride = cfg.conv7.stride,
            padding = cfg.conv7.padding,
            norm_layer= cfg.conv7.norm_layer,
            activation_layer = cfg.conv7.activation_layer
            )
        #pool
        self.classifier = nn.Sequential(
            #dense9
            nn.Linear(cfg.conv7.out_channels, cfg.dense9.hidden_dim),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            #dense10
            nn.Linear(cfg.dense9.hidden_dim, num_classes),
        )
        self.cgap= TemporalCGAvgPool3D()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv7(x)
        x = F.adaptive_avg_pool3d(x, (x.shape[2],1,1))
        x = self.cgap(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    
    @staticmethod
    def _clean_activation_buffers(m):
        if issubclass(type(m),CausalModule):
            m.reset_activation()

    def clean_activation_buffers(self) -> None:
        self.apply(self._clean_activation_buffers)        



