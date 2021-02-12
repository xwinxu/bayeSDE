"""Shape tests for layers.

To run, execute:
pytest torchengine/test_diffeq_layers.py
"""

import torch
from torch import nn
from torchsde._impl import diffeq_layers

from . import utils

batch_size = 16
d, o = 784, 100
c, h, w = 3, 32, 32
out_channels = 10
kernel_size = 3
stride = 2
padding = 2
output_padding = 1
dilation = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _make_linear_inputs():
    t, y = torch.randn((), device=device), torch.randn(size=(batch_size, d), device=device)
    return t, y


def _make_conv_inputs():
    t, y = torch.randn((), device=device), torch.randn(size=(batch_size, c, h, w), device=device)
    return t, y


def test_linear():
    t, y = _make_linear_inputs()
    layer = diffeq_layers.Linear(y.shape[1], o)
    params = layer.make_initial_params()
    out = layer(t, y, params)

    layer_ref = nn.Linear(d, o)
    out_ref = layer_ref(y)
    assert out.size() == out_ref.size()


def test_concat_linear():
    t, y = _make_linear_inputs()
    layer = diffeq_layers.ConcatLinear(y.shape[1], o)
    params = layer.make_initial_params()
    out = layer(t, y, params)

    layer_ref = nn.Linear(d + 1, o)
    out_ref = layer_ref(utils.channel_cat(t, y))
    assert out.size() == out_ref.size()


def test_concat_squash_linear():
    t, y = _make_linear_inputs()
    layer = diffeq_layers.ConcatSquashLinear(y.shape[1], o)
    params = layer.make_initial_params()
    out = layer(t, y, params)

    layer_ref = nn.Linear(d + 1, o)
    out_ref = layer_ref(torch.cat((t.expand_as(y[..., :1]), y), dim=-1))
    assert out.size() == out_ref.size()


def test_conv2d():
    t, y = _make_conv_inputs()
    layer = diffeq_layers.ConcatConv2d(
        in_channels=y.shape[1],
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation
    )
    params = layer.make_initial_params()
    out = layer(t, y, params)

    layer_ref = nn.Conv2d(
        in_channels=y.shape[1] + 1,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation
    )
    out_ref = layer_ref(utils.channel_cat(t, y))
    assert out.size() == out_ref.size()


def test_conv_transpose2d():
    t, y = _make_conv_inputs()
    layer = diffeq_layers.ConcatConvTranspose2d(
        in_channels=y.shape[1],
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation
    )
    params = layer.make_initial_params()
    out = layer(t, y, params)

    layer_ref = nn.ConvTranspose2d(
        in_channels=y.shape[1] + 1,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation
    )
    out_ref = layer_ref(utils.channel_cat(t, y))
    assert out.size() == out_ref.size()


def test_diffeq_sequential():
    t, y = _make_conv_inputs()
    net = diffeq_layers.DiffEqSequential(
        diffeq_layers.ConcatConv2d(
            in_channels=y.shape[1],
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        ),
        diffeq_layers.DiffEqWrapper(nn.Tanh()),
        diffeq_layers.ConcatConvTranspose2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation
        )
    )
    params = net.make_initial_params()
    out = net(t, y, params)

    conv = nn.Conv2d(
        in_channels=y.shape[1] + 1,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation
    )
    conv_transpose = nn.ConvTranspose2d(
        in_channels=out_channels + 1,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation
    )
    out_ref = conv(utils.channel_cat(t, y))
    out_ref = nn.Tanh()(out_ref)
    out_ref = conv_transpose(utils.channel_cat(t, out_ref))
    assert out.size() == out_ref.size()


def test_make_ode_block():
    t, y = _make_conv_inputs()
    ode_block = diffeq_layers.make_ode_k3_block(input_size=y.shape[1:])
    params = ode_block.make_initial_params()
    out = ode_block(t, y, params)
    assert out.size() == y.size()

    ode_block_with_squeeze = diffeq_layers.make_ode_k3_block(input_size=y.shape[1:], squeeze=True)
    params = ode_block_with_squeeze.make_initial_params()
    out = ode_block_with_squeeze(t, y, params)
    assert out.size() == (y.shape[0], y.shape[1] * 4, y.shape[2] // 2, y.shape[3] // 2)
