"""
Code inspired by:
https://pytorch.org/vision/stable/_modules/torchvision/models/mobilenetv2.html
https://pytorch.org/vision/stable/_modules/torchvision/models/mobilenetv3.html
"""
from collections import OrderedDict
import torch
from torch.nn.modules.utils import _triple, _pair
import torch.nn.functional as F
from typing import Any, Callable, Optional, Tuple, Union
from einops import rearrange
from torch import nn, Tensor

class HardSigmoid(torch.autograd.Function):
    #https://linuxtut.com/en/3e7495014f85eb6103fc/
    
    #TODO Legacy autograd function with non-static forward method is deprecated. Please use new-style autograd function with static forward method
    @staticmethod
    def forward(ctx, i):
         ctx.save_for_backward(i)
         result = (0.2 * i + 0.5).clamp(min=0.0, max=1.0)
         return result

    @staticmethod
    def backward(ctx, grad_output):
         grad_input = grad_output.clone()
         result, = ctx.saved_tensors
         grad_input *= 0.2
         grad_input[result < -2.5] = 0
         grad_input[result > -2.5] = 0
         return grad_input


class swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)


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

    def forward(self, x: Tensor) -> Tensor:
        input_shape = x.shape
        cumulative_sum = torch.cumsum(x, dim=2)
        if self.activation is None:
            self.activation = cumulative_sum[:,:,-1:].clone()#TODO check this
        else:
            cumulative_sum += self.activation
            self.activation += cumulative_sum[:,:,-1:]
        divisor = torch.range(1, input_shape[2])[None,None,:, None,None].expand(x.shape)

        x = cumulative_sum/(self.T+divisor)
        self.T += input_shape[2]
        print(self.T, self.activation)
        return x

    @staticmethod
    def _detach_activation(module: nn.Module,
                           input: Tensor,
                           output: Tensor) -> None:
        module.activation.detach_()

    def reset_activation(self) -> None:
        super().reset_activation()
        self.T = 0

class Conv2dBNActivation(nn.Sequential):
    def __init__(
                 self,
                 in_planes: int,
                 out_planes: int,
                 *,
                 kernel_size: Union[int, Tuple[int, int]],
                 padding: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None,
                 **kwargs: Any,
                 ) -> None:
        kernel_size = _pair(kernel_size)#change to duple
        stride = _pair(stride)
        padding = _pair(padding)
        convolution = nn.Conv2d
        if norm_layer is None:#TODO change this behaviour, norm can't bee an activation specific
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = swish
        self.kernel_size = kernel_size
        self.stride = stride
        dict_layers = OrderedDict({
                            "conv2d": convolution(in_planes, out_planes,
                                                  kernel_size=kernel_size,
                                                  stride=stride,
                                                  padding=padding,
                                                  groups=groups,
                                                  **kwargs),
                            "norm": norm_layer(out_planes, eps=0.001),
                            "act": activation_layer()
                            })

        self.out_channels = out_planes
        super(Conv2dBNActivation, self).__init__(dict_layers)

 

class CausalConv3D(CausalModule):
    def __init__(
            self,
            in_planes: int,
            out_planes: int,
            kernel_size: Union[int, Tuple[int, int, int]],
            padding: Union[int, Tuple[int, int]],
            conv_type,
            stride,#TODO typing
            **kwargs: Any,#TODO probably wrong typing
            ) -> None:
        super().__init__()
        kernel_size = _triple(kernel_size)
        padding = _pair(padding)
        self.kernel_size = kernel_size
        self.dim_pad = self.kernel_size[0]-1
        padding = (0, padding[0], padding[1])#TODO change this
        self.conv_type = conv_type
        self.conv_2= None
        #TODO check right convolution type in input
        if conv_type == "2plus1d":
            self.conv_1 = Conv2dBNActivation(in_planes,
                                  out_planes,
                                  kernel_size = (kernel_size[1],kernel_size[2]),
                                  padding=(padding[1],padding[2]),
                                  stride=(stride[1],stride[2]),
                                  **kwargs)
            if kernel_size[0]>1:
                self.conv_2 = Conv2dBNActivation(in_planes,
                                          out_planes,
                                          kernel_size = (kernel_size[0],1),
                                          padding=(padding[0],0),
                                          stride=(stride[0],1),#TODO check all this stride and paddings
                                          **kwargs)
        else:
            self.conv_1 = nn.Conv3d(in_planes,
                                  out_planes,
                                  kernel_size,
                                  padding=padding,
                                  stride=stride
                                  **kwargs)


    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        if self.dim_pad >0 and self.conv_2 is None:
            x = self._cat_stream_buffer(x, device)
        shape_with_buffer = x.shape
        if self.conv_type == "2plus1d":
            x = rearrange(x, "b c t h w -> (b t) c h w") 
        x = self.conv_1(x)
        if self.conv_type == "2plus1d":
            x = rearrange(x, "(b t) c h w -> b c t h w", t = shape_with_buffer[2])
            if self.dim_pad >0:
                x = self._cat_stream_buffer(x, device)

            if self.conv_2 is not None:
                x = rearrange(x,"b c t h w -> b c t (h w)") 
                x = self.conv_2(x)
                x = rearrange(x,"b c t (h w) -> b c t h w", h = int(x.shape[3]**(0.5))) #TODO this can cause problems
        return x

    def _cat_stream_buffer(self, x: Tensor, device ) -> Tensor:#TODO add typing
        if self.activation is None:
            self._setup_activation(x.shape)
        x = torch.cat((self.activation.to(device), x), 2)
        self._save_in_activation(x)
        return x

    def _save_in_activation(self, x: Tensor) -> Tensor:
        assert self.dim_pad > 0
        self.activation = x[:, :, -self.dim_pad:, ...].clone().detach()

    def _setup_activation(self, input_shape: Tuple[float, ...]) -> None:
        assert self.dim_pad > 0
        self.activation = torch.zeros(*input_shape[:2],
                                      self.dim_pad,
                                      *input_shape[3:])


class SqueezeExcitation(nn.Module):

    def __init__(self, input_channels: int,activation_2,  causal = False, 
            activation_1 = swish(),se_type= "3d" ,squeeze_factor: int = 4):
        super().__init__()
        self.se_type = se_type
        self.causal = causal#TODO this is somethign that we need to fix with 2plus3d
        se_multiplier = 2 if se_type == "2plus1d" else 1
        se_in_multiplier = 2 if se_type == "2plus1d" else 1
        squeeze_channels = _make_divisible(input_channels//squeeze_factor*se_multiplier, 8)
        self.temporal_cumualtive_GAvg3D = TemporalCGAvgPool3D()
        self.fc1 = nn.Conv3d(input_channels*se_in_multiplier, squeeze_channels, (1, 1, 1))
        self.activation_1 = activation_1
        self.activation_2 = activation_2
        self.fc2 = nn.Conv3d(squeeze_channels, input_channels, (1, 1, 1))

    def _scale(self, input: Tensor) -> Tensor:
        if self.causal:
            x_space = torch.mean(input, dim = [3,4],keepdim = True)
            scale = self.temporal_cumualtive_GAvg3D(x_space)
            scale = torch.cat((scale,x_space),dim = 1)
        else:
            scale = F.adaptive_avg_pool3d(input, 1)
        scale = self.fc1(scale)
        scale = self.activation_1(scale)
        scale = self.fc2(scale)
        return self.activation_2(scale)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        return scale * input


def _make_divisible(v: float,
                    divisor: int,
                    min_value: Optional[int] = None
                    ) -> int:
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


def same_padding(x: Tensor,
                 in_height: int, in_width: int,
                 stride_h: int, stride_w: int,
                 filter_height: int, filter_width: int) -> Tensor:
    if (in_height % stride_h == 0):
        pad_along_height = max(filter_height - stride_h, 0)
    else:
        pad_along_height = max(filter_height - (in_height % stride_h), 0)
    if (in_width % stride_w == 0):
        pad_along_width = max(filter_width - stride_w, 0)
    else:
        pad_along_width = max(filter_width - (in_width % stride_w), 0)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    padding_pad = (pad_left, pad_right, pad_top, pad_bottom)
    return torch.nn.functional.pad(x, padding_pad)


class tfAvgPool3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgf = nn.AvgPool3d((1, 3, 3), stride=(1, 2, 2))

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != x.shape[-2]:
            raise RuntimeError('only same shape for h and w are supported by avg with tf_like')
        if x.shape[-1] != x.shape[-2]:
            raise RuntimeError('only same shape for h and w are supported by avg with tf_like')
        f1 = x.shape[-1] % 2 != 0
        if f1:
            padding_pad = (0, 0, 0, 0)
        else:
            padding_pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, padding_pad)
        if f1:
            x = torch.nn.functional.avg_pool3d(x,
                                               (1, 3, 3),
                                               stride=(1, 2, 2),
                                               count_include_pad=False,
                                               padding=(0, 1, 1))
        else:
            x = self.avgf(x)
            x[..., -1] = x[..., -1] * 9/6
            x[..., -1, :] = x[..., -1, :] * 9/6
        return x


class Conv3DBNActivation(nn.Sequential):
    def __init__(
                 self,
                 in_planes: int,
                 out_planes: int,
                 *,
                 causal: bool,
                 tf_like: bool,
                 kernel_size: Union[int, Tuple[int, int, int]],
                 padding: Union[int, Tuple[int, int, int]],
                 conv_type: str,#TODO check correnctness
                 stride: Union[int, Tuple[int, int, int]] = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None,
                 ) -> None:
        self.tf_like = tf_like
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        if tf_like:
            if kernel_size[0] % 2 == 0:
                raise ValueError('tf_like supports only odd kernels for temporal dimension')
            padding = ((kernel_size[0]-1)//2, 0, 0)
            if stride[0] != 1:
                raise ValueError('illegal stride value, tf like supports only stride == 1 for temporal dimension')
            if stride[1] > kernel_size[1] or stride[2] > kernel_size[2]:
                raise ValueError('tf_like supports only stride <= of the kernel size')
        if causal:
            padding = (padding[1], padding[2])
            convolution = CausalConv3D
        else:
            convolution = nn.Conv3d
        if norm_layer is None:
            norm_layer = nn.Identity 
        if activation_layer is None:
            activation_layer = nn.Identity 
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv_type = conv_type
        if conv_type == "3d":
            dict_layers = OrderedDict({#TODO this cannot be causal and 3d at the same time
                                "conv3d": convolution(in_planes, out_planes,
                                                      kernel_size=kernel_size,
                                                      stride=stride,
                                                      padding=padding,
                                                      groups=groups,
                                                      bias=False),
                                "norm": norm_layer(out_planes, eps=0.001),
                                "act": activation_layer()
                                })
        if conv_type == "2plus1d":
            dict_layers = OrderedDict({#TODO 2plus1d convolution work only on causal mode
                                "2plus1s": convolution(in_planes, out_planes,
                                                      kernel_size=kernel_size,
                                                      stride=stride,
                                                      padding=padding,
                                                      groups=groups,
                                                      conv_type = conv_type,
                                                      activation_layer = activation_layer ,
                                                      norm_layer = norm_layer, 
                                                      bias=False),
                                })


        super(Conv3DBNActivation, self).__init__(dict_layers)
        self.out_channels = out_planes

    def forward(self, x):
        if self.tf_like:
            x = same_padding(x, x.shape[-2], x.shape[-1],
                             self.stride[-2], self.stride[-1],
                             self.kernel_size[-2], self.kernel_size[-1])
        return super().forward(x)

#TODO check all typings
class BasicBneck(nn.Module):
    def __init__(self, cfg: "CfgNode", causal: bool, tf_like: bool, conv_type ) -> None:
        super().__init__()
        assert type(cfg.stride) is tuple
        if not cfg.stride[0] == 1 or not (1 <= cfg.stride[1] <= 2) or not (1 <= cfg.stride[2] <= 2):
            raise ValueError('illegal stride value')
        self.res = None
        layers = []
        if cfg.expanded_channels != cfg.out_channels:
            # expand
            self.expand = Conv3DBNActivation(
                in_planes=cfg.input_channels,
                out_planes=cfg.expanded_channels,
                kernel_size=(1, 1, 1),
                padding=(0, 0, 0),
                causal=causal,
                conv_type=conv_type,
                tf_like=tf_like,
                norm_layer=nn.BatchNorm3d if conv_type == "3d" else nn.BatchNorm2d,
                activation_layer=swish if conv_type == "3d" else nn.Hardswish
                )
        # deepwise
        self.deep = Conv3DBNActivation(
            in_planes=cfg.expanded_channels,
            out_planes=cfg.expanded_channels,
            kernel_size=cfg.kernel_size,
            padding=cfg.padding,
            stride=cfg.stride,
            groups=cfg.expanded_channels,
            causal=causal,
            conv_type=conv_type,
            tf_like=tf_like,
            norm_layer=nn.BatchNorm3d if conv_type == "3d" else nn.BatchNorm2d,
            activation_layer=swish if conv_type == "3d" else nn.Hardswish
            )
        # SE
        self.se = SqueezeExcitation(cfg.expanded_channels, 
                causal = causal, activation_1 = swish() if conv_type == "3d" else nn.Hardswish(),
                activation_2 = torch.sigmoid if conv_type == "3d" else HardSigmoid().apply, 
                se_type = conv_type
                )
        # project
        self.project = Conv3DBNActivation(
            cfg.expanded_channels,
            cfg.out_channels,
            kernel_size=(1, 1, 1),
            padding=(0, 0, 0),
            causal=causal,
            conv_type=conv_type,
            tf_like=tf_like,
            norm_layer=nn.BatchNorm3d if conv_type == "3d" else nn.BatchNorm2d,
            activation_layer=nn.Identity
            )

        if not (cfg.stride == (1, 1, 1) and cfg.input_channels == cfg.out_channels):
            if cfg.stride != (1, 1, 1):
                if tf_like:

                    layers.append(tfAvgPool3D())
                else:
                    layers.append(nn.AvgPool3d((1, 3, 3),
                                  stride=cfg.stride,
                                  padding=cfg.padding_avg))
            layers.append(Conv3DBNActivation(
                    in_planes=cfg.input_channels,
                    out_planes=cfg.out_channels,
                    kernel_size=(1, 1, 1),
                    padding=(0, 0, 0),
                    norm_layer=nn.BatchNorm3d if conv_type == "3d" else nn.BatchNorm2d,
                    activation_layer=nn.Identity,
                    causal=causal,
                    conv_type=conv_type,
                    tf_like=tf_like
                    ))
            self.res = nn.Sequential(*layers)
        # ReZero
        self.alpha = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, input: Tensor) -> Tensor:
        result = input
        if self.res is not None:
            input = self.res(input)
        if self.expand is not None:
            result = self.expand(result)
        result = self.deep(result)
        result = self.se(result)
        result = self.project(result)
        result = input + self.alpha * result
        return result


class MoViNet(nn.Module):
    def __init__(self,
                 cfg: "CfgNode",
                 num_classes: int,
                 causal: bool = True,
                 pretrained: bool = False,
                 tf_like: bool = False
                 ) -> None:
        super().__init__()
        if pretrained:
            assert causal is False, "weights are only available for non causal models"
            assert tf_like, "pretained model available only with tf_like behavior"
        if causal is True:
            conv_type = "2plus1d"
        else:
            conv_type = "3d"
        blocks_dic = OrderedDict()
        # conv1
        self.conv1 = Conv3DBNActivation(
            in_planes=cfg.conv1.input_channels,
            out_planes=cfg.conv1.out_channels,
            kernel_size=cfg.conv1.kernel_size,
            stride=cfg.conv1.stride,
            padding=cfg.conv1.padding,
            causal=causal,
            conv_type=conv_type,
            tf_like=tf_like,
            norm_layer=nn.BatchNorm3d if conv_type == "3d" else nn.BatchNorm2d,
            activation_layer=swish if conv_type == "3d" else nn.Hardswish
            )
        # blocks
        for i, block in enumerate(cfg.blocks):
            for j, basicblock in enumerate(block):
                blocks_dic[f"b{i}_l{j}"] = BasicBneck(basicblock,
                                                      causal=causal,
                                                      conv_type=conv_type,
                                                      tf_like=tf_like)
        self.blocks = nn.Sequential(blocks_dic)
        # conv7
        self.conv7 = Conv3DBNActivation(
            in_planes=cfg.conv7.input_channels,
            out_planes=cfg.conv7.out_channels,
            kernel_size=cfg.conv7.kernel_size,
            stride=cfg.conv7.stride,
            padding=cfg.conv7.padding,
            causal=causal,
            conv_type=conv_type,
            tf_like=tf_like,
            norm_layer=nn.BatchNorm3d if conv_type == "3d" else nn.BatchNorm2d,
            activation_layer=swish if conv_type == "3d" else nn.Hardswish
            )
        # pool
        self.classifier = nn.Sequential(
            # dense9
            nn.Conv3d(cfg.conv7.out_channels,  cfg.dense9.hidden_dim, (1, 1, 1)),
            swish(),#TODO
            nn.Dropout(p=0.2, inplace=True),
            # dense10
            nn.Conv3d(cfg.dense9.hidden_dim,  num_classes, (1, 1, 1)),
        )
        if causal:
            self.cgap = TemporalCGAvgPool3D()
        if pretrained:
            state_dict = torch.hub.load_state_dict_from_url(cfg.weights)
            self.load_state_dict(state_dict)
        else:
            self.apply(self._weight_init)
        self.causal = causal

    def avg(self, x):
        if self.causal:
            avg = F.adaptive_avg_pool3d(x, (x.shape[2], 1, 1))
            avg = self.cgap(avg)[:,:,-1:]
        else:
            avg = F.adaptive_avg_pool3d(x, 1)
        return avg

    @staticmethod
    def _weight_init(m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv7(x)
        x = self.avg(x)
        x = self.classifier(x)
        x = x.flatten(1)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    @staticmethod
    def _clean_activation_buffers(m):
        if issubclass(type(m), CausalModule):
            m.reset_activation()

    def clean_activation_buffers(self) -> None:
        self.apply(self._clean_activation_buffers)
