"""Differential equation layers that support explicitly 
passing parameters as tensors to `forward`.
"""

import abc
import logging
from typing import Any, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair

from . import utils

_shape_t = Union[int, List[int], torch.Size]

class Augment(nn.Module):
    def __init__(self, aug_dim):
        super(Augment, self).__init__()
        self.aug_dim = aug_dim
    
    def forward(self, y, *args, **kwargs):
        z = torch.zeros(y.shape[:-1] + (1 * self.aug_dim,))
        aug_z = torch.cat((y, z), axis=1) # channel dim = 1
        return aug_z


class DiffEqModule(abc.ABC, nn.Module):
    def make_initial_params(self):
        return [p.detach().clone() for p in self.parameters()]

    # Fixes stupid pylint bug due to 1.6.0:
    # https://github.com/pytorch/pytorch/issues/42305
    def _forward_unimplemented(self, *input: Any) -> None:
        pass


class DiffEqSequential(DiffEqModule):
    """Entry point for building drift on hidden state of neural network."""

    def __init__(self, *layers: DiffEqModule, explicit_params=True):
        # explicit_params: Register the built-in parameters if True; else use `params` argument of `forward`.
        super(DiffEqSequential, self).__init__()
        self.layers = layers if explicit_params else nn.ModuleList(layers)
        self.explicit_params = explicit_params

    def forward(self, t, y, params: Optional[List] = None):
        if params is None:
            for layer in self.layers:
                y = layer(t, y)
        else:
            for layer, params_ in zip(self.layers, params):
                y = layer(t, y, params_)
        return y

    def make_initial_params(self):
        return [layer.make_initial_params() for layer in self.layers]

    def __repr__(self):
        return repr(nn.Sequential(*self.layers)) if self.explicit_params else repr(self.layers)


class DiffEqWrapper(DiffEqModule):
    def __init__(self, module):
        super(DiffEqWrapper, self).__init__()
        self.module = module

    def forward(self, t, y, *args, **kwargs):
        del t, args, kwargs
        return self.module(y)


class ConcatConv2d(nn.Conv2d, DiffEqModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: _size_2_t = 0,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros'):
        super(ConcatConv2d, self).__init__(
            in_channels=in_channels + 1,  # Extra time channel!
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode
        )
        # Make this selection a one-time cost, as opposed to putting if-else checks in `forward`.
        self.unpack_params = self.unpack_wb if bias else self.unpack_w

    def forward(self, t, y, params: Optional[List] = None):
        weight, bias = self.unpack_params(params)  # noqa
        ty = utils.channel_cat(t, y)

        if self.padding_mode != 'zeros':
            return F.conv2d(
                F.pad(ty, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                weight, bias, self.stride, _pair(0), self.dilation, self.groups
            )
        return F.conv2d(ty, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def unpack_wb(self, params: Optional[List] = None):  # noqa
        if params is None:
            return self.weight, self.bias
        return params

    def unpack_w(self, params: Optional[List] = None):  # noqa
        if params is None:
            return self.weight, self.bias
        return params[0], None  # Bias is None.


class ConcatConvTranspose2d(nn.ConvTranspose2d, DiffEqModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: _size_2_t = 0,
                 output_padding: _size_2_t = 0,
                 groups: int = 1,
                 bias: bool = True,
                 dilation: int = 1,
                 padding_mode: str = 'zeros'):
        super(ConcatConvTranspose2d, self).__init__(
            in_channels=in_channels + 1,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode
        )
        self.unpack_params = self.unpack_wb if bias else self.unpack_w

    def forward(self, t, y, params: Optional[List] = None, output_size: Optional[List[int]] = None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

        weight, bias = self.unpack_params(params)  # noqa
        ty = utils.channel_cat(t, y)
        output_padding = self._output_padding(ty, output_size, self.stride, self.padding, self.kernel_size)  # noqa
        return F.conv_transpose2d(
            ty, weight, bias, self.stride, self.padding, output_padding, self.groups, self.dilation)

    def unpack_wb(self, params: Optional[List] = None):  # noqa
        if params is None:
            return self.weight, self.bias
        return params

    def unpack_w(self, params: Optional[List] = None):  # noqa
        if params is None:
            return self.weight, self.bias
        return params[0], None  # Bias is None.


class Linear(nn.Linear, DiffEqModule):
    def __init__(self, in_features: int, out_features: int):
        super(Linear, self).__init__(in_features=in_features, out_features=out_features)

    def forward(self, t, y, params: Optional[List] = None):
        w, b = (self.weight, self.bias) if params is None else params
        return F.linear(y, w, b)


class ConcatLinear(nn.Linear, DiffEqModule):
    def __init__(self, in_features: int, out_features: int):
        super(ConcatLinear, self).__init__(
            in_features=in_features + 1,  # Extra time channel!
            out_features=out_features,
        )

    def forward(self, t, y, params: Optional[List] = None):
        w, b = (self.weight, self.bias) if params is None else params
        ty = utils.channel_cat(t, y)
        return F.linear(ty, w, b)


class ConcatSquashLinear(nn.Linear, DiffEqModule):
    def __init__(self, in_features: int, out_features: int):
        super(ConcatSquashLinear, self).__init__(
            in_features=in_features + 1,  # Extra time channel!
            out_features=out_features
        )
        self.wt = nn.Parameter(torch.randn(1, self.out_features))
        self.by = nn.Parameter(torch.randn(1, self.out_features))
        self.b = nn.Parameter(torch.randn(1, self.out_features))

    def forward(self, t, y, params: Optional[List] = None):
        wy, by, wt, bt, b = (self.weight, self.bias, self.wt, self.bt, self.b) if params is None else params
        ty = utils.channel_cat(t, y)

        net = F.linear(ty, wy, by)
        scale = torch.sigmoid(t * wt + bt)
        shift = b * t
        return scale * net + shift

    def make_initial_params(self):
        return [
            self.weight.clone(),  # wy
            self.bias.clone(),  # by
            torch.randn(1, self.out_features),  # wt
            torch.randn(1, self.out_features),  # bt
            torch.randn(1, self.out_features),  # b
        ]


class SqueezeDownsample(DiffEqModule):
    def __init__(self):
        super(SqueezeDownsample, self).__init__()

    def forward(self, t, y, *args, **kwargs):  # noqa
        del t, args, kwargs
        b, c, h, w = y.size()
        return y.reshape(b, c * 4, h // 2, w // 2)


class ConvDownsample(ConcatConv2d):
    def __init__(self, input_size):
        c, h, w = input_size
        super(ConvDownsample, self).__init__(
            in_channels=c,
            out_channels=c * 4,
            kernel_size=3,
            stride=2,
            padding=1
        )


class LayerNormalization(nn.LayerNorm, DiffEqModule):
    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True):
        super(LayerNormalization, self).__init__(
            normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, t, y, params: Optional[List] = None):  # noqa
        del t
        weight, bias = (self.weight, self.bias) if params is None else params
        return F.layer_norm(y, self.normalized_shape, weight, bias, self.eps)


class Print(DiffEqModule):
    def __init__(self, name=None):
        super(Print, self).__init__()
        self.name = name

    def forward(self, t, y, *args, **kwargs):
        del t, args, kwargs
        msg = (
            f"size: {y.size()}, "
            f"mean: {y.mean()}, "
            f"abs mean: {y.abs().mean()}, "
            f"max: {y.max()}, "
            f"abs max: {y.abs().max()}, "
            f"min: {y.min()}, "
            f"abs min: {y.abs().min()}"
        )
        if self.name is not None:
            msg = f"{self.name}, " + msg
        logging.warning(msg)
        return y


def make_ode_k3_block(input_size, activation="softplus", squeeze=False):
    """Make a block of kernel size 3 for all convolutions."""
    return DiffEqSequential(*make_ode_k3_block_layers(input_size, activation, squeeze))


def make_ode_k3_block_layers(input_size,
                             activation="softplus",
                             squeeze=False,
                             last_activation=True,
                             hidden_width=128,
                             mode=0):
    channels, height, width = input_size
    activation = utils.select_activation(activation)
    if mode == 0:
        layers = [
            ConcatConv2d(in_channels=channels, out_channels=hidden_width, kernel_size=3, padding=1),
            DiffEqWrapper(activation()),
            ConcatConv2d(in_channels=hidden_width, out_channels=hidden_width, kernel_size=3, stride=2, padding=1),
            DiffEqWrapper(activation()),
            ConcatConvTranspose2d(
                in_channels=hidden_width, out_channels=hidden_width, kernel_size=3, stride=2, output_padding=1),
            DiffEqWrapper(activation()),
            ConcatConv2d(in_channels=hidden_width, out_channels=channels, kernel_size=3),
        ]
    else:
        layers = [
            ConcatConv2d(in_channels=channels, out_channels=hidden_width, kernel_size=3, padding=1),
            DiffEqWrapper(activation()),
            ConcatConv2d(in_channels=hidden_width, out_channels=hidden_width, kernel_size=3, padding=1),
            DiffEqWrapper(activation()),
            ConcatConv2d(in_channels=hidden_width, out_channels=channels, kernel_size=3, padding=1),
        ]
    if last_activation:
        layers.extend([DiffEqWrapper(activation())])
    if squeeze:
        layers.extend([ConvDownsample(input_size=input_size)])
    return layers
