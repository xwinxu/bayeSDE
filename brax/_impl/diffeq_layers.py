"""
Implementations of different time-dependent ODE/SDE layers.
"""

import jax
import jax.numpy as np
from jax import random
from jax.experimental import stax
from jax.nn.initializers import glorot_normal, normal, he_normal


def shape_dependent(make_layer):
    """Replaces stax.shape_dependent for time dependent inputs
    Args:
      make_layer: a one-argument function that takes an input shape as an argument
        (a tuple of positive integers) and returns an (init_fun, apply_fun) pair.
    Returns:
      A new layer, meaning an (init_fun, apply_fun) pair, representing the same
      layer as returned by `make_layer` but with its construction delayed until
      input shapes are known.
    """
    def init_fun(rng, input_shape):
        return make_layer(input_shape)[0](rng, input_shape)

    def apply_fun(params, inputs, **kwargs):
        # x = inputs
        x, t = inputs
        return make_layer(x.shape)[1](params, (x, t), **kwargs)

    return init_fun, apply_fun


def IgnoreLinear(out_dim, W_init=he_normal(), b_init=normal()):
    """ y = Wx + b
    """
    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        k1, k2 = random.split(rng)
        W, b = W_init(k1, (input_shape[-1], out_dim)), b_init(k2, (out_dim,))
        return output_shape, (W, b)

    def apply_fun(params, inputs, **kwargs):
        x, t = inputs
        W, b = params
        return (np.dot(x, W) + b, t)
    return init_fun, apply_fun


def ConcatLinear(out_dim, W_init=he_normal(), b_init=normal()):
    """ y = Wx + b + at
    """
    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        k1, k2 = random.split(rng)
        W, b = W_init(k1, (input_shape[-1] + 1,
                           out_dim)), b_init(k2, (out_dim,))
        return output_shape, (W, b)

    def apply_fun(params, inputs, **kwargs):
        x, t = inputs
        W, b = params

        # concatenate t onto the inputs
        tt = t.reshape([-1] * (x.ndim - 1) + [1])  # single batch example
        # i.e. [:, :, ..., :, :1] column vector
        tt = np.tile(tt, x.shape[:-1] + (1,))
        xtt = np.concatenate([x, tt], axis=-1)

        return (np.dot(xtt, W) + b, t)
    return init_fun, apply_fun


def ConcatSquashLinear(out_dim, W_init=he_normal(), b_init=normal()):
    """ y = Sigmoid(at + c)(Wx + b) + dt. Note: he_normal only takes multi dim.
    """
    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        k1, k2, k3, k4, k5 = random.split(rng, 5)
        W, b = W_init(k1, (input_shape[-1], out_dim)), b_init(k2, (out_dim,))
        w_t, w_tb = b_init(k3, (out_dim,)), b_init(k4, (out_dim,))
        b_t = b_init(k5, (out_dim,))
        return output_shape, (W, b, w_t, w_tb, b_t)

    def apply_fun(params, inputs, **kwargs):
        x, t = inputs
        W, b, w_t, w_tb, b_t = params

        # (W.xtt + b) *
        out = np.dot(x, W) + b
        # sigmoid(a.t + c)  +
        out *= jax.nn.sigmoid(w_t * t + w_tb)
        # d.t
        out += b_t * t

        return (out, t)
    return init_fun, apply_fun


# dimension_numbers = ('NCHW', 'OIHW', 'NCHW') # torch version
dimension_numbers = ('NHWC', 'HWIO', 'NHWC')


def IgnoreConv2D(out_dim, W_init=he_normal(), b_init=normal(), kernel=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
    assert dilation == 1 and groups == 1
    if not transpose:
        init_fun_wrapped, apply_fun_wrapped = stax.GeneralConv(
            dimension_numbers, out_chan=out_dim, filter_shape=(
                kernel, kernel), strides=(stride, stride), padding=padding
        )
    else:
        init_fun_wrapped, apply_fun_wrapped = stax.GeneralConvTranspose(
            dimension_numbers, out_chan=out_dim, filter_shape=(
                kernel, kernel), strides=(stride, stride), padding=padding
        )

    def apply_fun(params, inputs, **kwargs):
        x, t = inputs
        out = apply_fun_wrapped(params, x, **kwargs)
        return (out, t)

    return init_fun_wrapped, apply_fun_wrapped


def ConcatConv2D(out_dim, W_init=he_normal(), b_init=normal(), kernel=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
    assert dilation == 1 and groups == 1
    if not transpose:
        init_fun_wrapped, apply_fun_wrapped = stax.GeneralConv(
            dimension_numbers, out_chan=out_dim, filter_shape=(
                kernel, kernel), strides=(stride, stride), padding=padding
        )
    else:
        init_fun_wrapped, apply_fun_wrapped = stax.GeneralConvTranspose(
            dimension_numbers, out_chan=out_dim, filter_shape=(
                kernel, kernel), strides=(stride, stride), padding=padding
        )

    def init_fun(rng, input_shape):  # note, input shapes only take x
        concat_input_shape = list(input_shape)
        concat_input_shape[-1] += 1    # add time channel dim
        concat_input_shape = tuple(concat_input_shape)
        return init_fun_wrapped(rng, concat_input_shape)

    def apply_fun(params, inputs, **kwargs):
        x, t = inputs
        tt = np.ones_like(x[:, :, :, :1]) * t
        xtt = np.concatenate([x, tt], axis=-1)
        out = apply_fun_wrapped(params, xtt, **kwargs)
        return (out, t)

    return init_fun, apply_fun


def ConcatConv2D_v2(out_dim, W_init=he_normal(), b_init=normal(), kernel=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
    assert dilation == 1 and groups == 1
    if not transpose:
        init_fun_wrapped, apply_fun_wrapped = stax.GeneralConv(
            dimension_numbers, out_chan=out_dim, filter_shape=(
                kernel, kernel), strides=(stride, stride), padding=padding
        )
    else:
        init_fun_wrapped, apply_fun_wrapped = stax.GeneralConvTranspose(
            dimension_numbers, out_chan=out_dim, filter_shape=(
                kernel, kernel), strides=(stride, stride), padding=padding
        )

    def init_fun(rng, input_shape):
        k1, k2 = random.split(rng)
        output_shape_conv, params_conv = init_fun_wrapped(k1, input_shape)
        W_hyper_bias = W_init(k2, (1, out_dim))

        return output_shape_conv, (params_conv, W_hyper_bias)

    def apply_fun(params, inputs, **kwargs):
        x, t = inputs
        params_conv, W_hyper_bias = params
        out = apply_fun_wrapped(params_conv, x, **kwargs) + np.dot(t.view(1, 1),
                                                                   W_hyper_bias).view(1, 1, 1, -1)  # if ncwh stead of nhwc: .view(1, -1, 1, 1)
        return (out, t)

    return init_fun, apply_fun


def ConcatSquashConv2D(out_dim, W_init=he_normal(), b_init=normal(), kernel=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
    assert dilation == 1 and groups == 1
    if not transpose:
        init_fun_wrapped, apply_fun_wrapped = stax.GeneralConv(
            dimension_numbers, out_chan=out_dim, filter_shape=(
                kernel, kernel), strides=(stride, stride), padding=padding
        )
    else:
        init_fun_wrapped, apply_fun_wrapped = stax.GeneralConvTranspose(
            dimension_numbers, out_chan=out_dim, filter_shape=(
                kernel, kernel), strides=(stride, stride), padding=padding
        )

    def init_fun(rng, input_shape):
        k1, k2, k3, k4 = random.split(rng, 4)
        output_shape_conv, params_conv = init_fun_wrapped(k1, input_shape)
        W_hyper_gate, b_hyper_gate = W_init(
            k2, (1, out_dim)), b_init(k3, (out_dim,))
        W_hyper_bias = W_init(k4, (1, out_dim))
        return output_shape_conv, (params_conv, W_hyper_gate, b_hyper_gate, W_hyper_bias)

    def apply_fun(params, inputs, **kwargs):
        x, t = inputs
        params_conv, W_hyper_gate, b_hyper_gate, W_hyper_bias = params
        conv_out = apply_fun_wrapped(params_conv, x, **kwargs)
        gate_out = jax.nn.sigmoid(
            np.dot(t.view(1, 1), W_hyper_gate) + b_hyper_gate).view(1, 1, 1, -1)
        bias_out = np.dot(t.view(1, 1), W_hyper_bias).view(1, 1, 1, -1)
        out = conv_out * gate_out + bias_out
        return (out, t)

    return init_fun, apply_fun


def ConcatCoordConv2D(out_dim, W_init=he_normal(), b_init=normal(), kernel=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
    assert dilation == 1 and groups == 1
    if not transpose:
        init_fun_wrapped, apply_fun_wrapped = stax.GeneralConv(
            dimension_numbers, out_chan=out_dim, filter_shape=(
                kernel, kernel), strides=(stride, stride), padding=padding
        )
    else:
        init_fun_wrapped, apply_fun_wrapped = stax.GeneralConvTranspose(
            dimension_numbers, out_chan=out_dim, filter_shape=(
                kernel, kernel), strides=(stride, stride), padding=padding
        )

    def init_fun(rng, input_shape):
        concat_input_shape = list(input_shape)
        # add time and coord channels; from 1 (torch) -> 0
        concat_input_shape[-1] += 3
        concat_input_shape = tuple(concat_input_shape)
        return init_fun_wrapped(rng, concat_input_shape)

    def apply_fun(params, inputs, **kwargs):
        x, t = inputs
        b, h, w, c = x.shape
        hh = np.arange(h).view(1, h, 1, 1).expand(b, h, w, 1)
        ww = np.arange(w).view(1, 1, w, 1).expand(b, h, w, 1)
        tt = t.view(1, 1, 1, 1).expand(b, h, w, 1)
        x_aug = np.concatenate([x, hh, ww, tt], axis=-1)
        out = apply_fun_wrapped(params, x_aug, **kwargs)
        return (out, t)

    return init_fun, apply_fun


def GatedConv2D(out_dim, W_init=he_normal(), b_init=normal(), kernel=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
    assert dilation == 1 and groups == 1
    if not transpose:
        init_fun_wrapped, apply_fun_wrapped = stax.GeneralConv(
            dimension_numbers, out_chan=out_dim, filter_shape=(
                kernel, kernel), strides=(stride, stride), padding=padding
        )
    else:
        init_fun_wrapped, apply_fun_wrapped = stax.GeneralConvTranspose(
            dimension_numbers, out_chan=out_dim, filter_shape=(
                kernel, kernel), strides=(stride, stride), padding=padding
        )

    def init_fun(rng, input_shape):
        k1, k2 = random.split(rng)
        output_shape, params_f = init_fun_wrapped(k1, input_shape)
        _, params_g = init_fun_wrapped(k2, input_shape)
        return output_shape, (params_f, params_g)

    def apply_fun(params, inputs, **kwargs):
        params_f, params_g = params
        f = apply_fun_wrapped(params_f, inputs)
        g = jax.nn.sigmoid(apply_fun_wrapped(params_g, inputs))
        return f * g

    return init_fun, apply_fun


def BlendConv2D(out_dim, W_init=he_normal(), b_init=normal(), kernel=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
    assert dilation == 1 and groups == 1
    if not transpose:
        init_fun_wrapped, apply_fun_wrapped = stax.GeneralConv(
            dimension_numbers, out_chan=out_dim, filter_shape=(
                kernel, kernel), strides=(stride, stride), padding=padding
        )
    else:
        init_fun_wrapped, apply_fun_wrapped = stax.GeneralConvTranspose(
            dimension_numbers, out_chan=out_dim, filter_shape=(
                kernel, kernel), strides=(stride, stride), padding=padding
        )

    def init_fun(rng, input_shape):
        k1, k2 = random.split(rng)
        output_shape, params_f = init_fun_wrapped(k1, input_shape)
        _, params_g = init_fun_wrapped(k2, input_shape)
        return output_shape, (params_f, params_g)

    def apply_fun(params, inputs, **kwargs):
        x, t = inputs
        params_f, params_g = params
        f = apply_fun_wrapped(params_f, x)
        g = apply_fun_wrapped(params_g, x)
        out = f + (g - f) * t
        return (out, t)

    return init_fun, apply_fun


def DiffEqWrapper(layer):
    """Wrapper for time dependent layers
    """
    init_fun, layer_apply_fun = layer

    def apply_fun(params, inputs, **kwargs):
        x, t = inputs
        return layer_apply_fun(params, x, **kwargs), t
    return init_fun, apply_fun


if __name__ == "__main__":

    init_fn, apply_fn = stax.serial(
        IgnoreLinear(20),
        DiffEqWrapper(stax.Relu),
        IgnoreLinear(20),
        DiffEqWrapper(stax.Relu)
    )

    key = random.PRNGKey(0)
    out_shape, params = init_fn(key, (-1, 2))  # includes the batch dimension

    x = random.normal(key, (5, 2))
    t = np.array(0.5)
    t, out = apply_fn(params, (x, t))

    print(out.shape)
    print(t.shape)
