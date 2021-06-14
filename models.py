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
from torch import nn, Tensor


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
        self.T += x.shape[2]
        if self.activation is None:
            self.activation = torch.sum(x, dim=2, keepdim=True)
        else:
            self.activation += torch.sum(x, dim=2, keepdim=True)

        x = self.activation/self.T
        return x

    @staticmethod
    def _detach_activation(module: nn.Module,
                           input: Tensor,
                           output: Tensor) -> None:
        module.activation.detach_()

    def reset_activation(self) -> None:
        super().reset_activation()
        self.T = 0


class CausalConv3D(CausalModule):
    def __init__(
            self,
            in_planes: int,
            out_planes: int,
            kernel_size: Union[int, Tuple[int, int, int]],
            padding: Union[int, Tuple[int, int]],
            **kwargs: Any,
            ) -> None:
        super().__init__()
        kernel_size = _triple(kernel_size)
        padding = _pair(padding)
        self.kernel_size = kernel_size
        self.dim_pad = self.kernel_size[0]-1
        padding = (0, padding[0], padding[1])
        self.conv = nn.Conv3d(in_planes,
                              out_planes,
                              kernel_size,
                              padding=padding,
                              **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        if self.dim_pad <= 0:
            x = self.conv(x)
        else:
            if self.activation is None:
                self._setup_activation(x.shape)
            x = torch.cat((self.activation.to(device), x), 2)
            self._save_in_activation(x)
            x = self.conv(x)
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

    def __init__(self, input_channels: int, squeeze_factor: int = 4):
        super().__init__()
        squeeze_channels = _make_divisible(input_channels//squeeze_factor, 8)
        self.fc1 = nn.Conv3d(input_channels, squeeze_channels, (1, 1, 1))
        self.swish = swish()
        self.fc2 = nn.Conv3d(squeeze_channels, input_channels, (1, 1, 1))

    def _scale(self, input: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool3d(input, 1)
        scale = self.fc1(scale)
        scale = self.swish(scale)
        scale = self.fc2(scale)
        return torch.sigmoid(scale)

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
            norm_layer = nn.BatchNorm3d
        if activation_layer is None:
            activation_layer = swish
        self.kernel_size = kernel_size
        self.stride = stride
        dict_layers = OrderedDict({
                            "conv3d": convolution(in_planes, out_planes,
                                                  kernel_size=kernel_size,
                                                  stride=stride,
                                                  padding=padding,
                                                  groups=groups,
                                                  bias=False),
                            "norm": norm_layer(out_planes, eps=0.001),
                            "act": activation_layer()
                            })
        super(Conv3DBNActivation, self).__init__(dict_layers)
        self.out_channels = out_planes

    def forward(self, x):
        if self.tf_like:
            x = same_padding(x, x.shape[-2], x.shape[-1],
                             self.stride[-2], self.stride[-1],
                             self.kernel_size[-2], self.kernel_size[-1])
        return super().forward(x)


class BasicBneck(nn.Module):
    def __init__(self, cfg: "CfgNode", causal: bool, tf_like: bool) -> None:
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
                norm_layer=cfg.norm_layer,
                activation_layer=cfg.activation_layer,
                causal=causal,
                tf_like=tf_like
                )
        # deepwise
        self.deep = Conv3DBNActivation(
            in_planes=cfg.expanded_channels,
            out_planes=cfg.expanded_channels,
            kernel_size=cfg.kernel_size,
            padding=cfg.padding,
            stride=cfg.stride,
            norm_layer=cfg.norm_layer,
            groups=cfg.expanded_channels,
            activation_layer=cfg.activation_layer,
            causal=causal,
            tf_like=tf_like
            )
        # SE
        self.se = SqueezeExcitation(cfg.expanded_channels)
        # project
        self.project = Conv3DBNActivation(
            cfg.expanded_channels,
            cfg.out_channels,
            kernel_size=(1, 1, 1),
            padding=(0, 0, 0),
            norm_layer=cfg.norm_layer,
            activation_layer=nn.Identity,
            causal=causal,
            tf_like=tf_like
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
                    norm_layer=cfg.norm_layer,
                    activation_layer=nn.Identity,
                    causal=causal,
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

        blocks_dic = OrderedDict()
        # conv1
        self.conv1 = Conv3DBNActivation(
            in_planes=cfg.conv1.input_channels,
            out_planes=cfg.conv1.out_channels,
            kernel_size=cfg.conv1.kernel_size,
            stride=cfg.conv1.stride,
            norm_layer=cfg.conv1.norm_layer,
            padding=cfg.conv1.padding,
            activation_layer=cfg.conv1.activation_layer,
            causal=causal,
            tf_like=tf_like
            )
        # blocks
        for i, block in enumerate(cfg.blocks):
            for j, basicblock in enumerate(block):
                blocks_dic[f"b{i}_l{j}"] = BasicBneck(basicblock,
                                                      causal=causal,
                                                      tf_like=tf_like)
        self.blocks = nn.Sequential(blocks_dic)
        # conv7
        self.conv7 = Conv3DBNActivation(
            in_planes=cfg.conv7.input_channels,
            out_planes=cfg.conv7.out_channels,
            kernel_size=cfg.conv7.kernel_size,
            stride=cfg.conv7.stride,
            padding=cfg.conv7.padding,
            norm_layer=cfg.conv7.norm_layer,
            activation_layer=cfg.conv7.activation_layer,
            causal=causal,
            tf_like=tf_like
            )
        # pool
        self.classifier = nn.Sequential(
            # dense9
            nn.Conv3d(cfg.conv7.out_channels,  cfg.dense9.hidden_dim, (1, 1, 1)),
            swish(),
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
            avg = self.cgap(avg)
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
