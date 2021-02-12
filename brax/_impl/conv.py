import functools
import itertools

import jax.numpy as np
from jax import lax, random
from jax.nn.initializers import glorot_normal, normal


def GeneralConv(dimension_numbers, out_chan, filter_shape,
                strides=None, padding='VALID', W_init=None,
                b_init=normal(1e-6), bias=True):
    """Layer construction function for a general convolution layer."""
    lhs_spec, rhs_spec, out_spec = dimension_numbers
    one = (1,) * len(filter_shape)
    strides = strides or one
    W_init = W_init or glorot_normal(rhs_spec.index('I'), rhs_spec.index('O'))

    def init_fun(rng, input_shape):
        filter_shape_iter = iter(filter_shape)
        kernel_shape = [out_chan if c == 'O' else
                        input_shape[lhs_spec.index('C')] if c == 'I' else
                        next(filter_shape_iter) for c in rhs_spec]
        output_shape = lax.conv_general_shape_tuple(
            input_shape, kernel_shape, strides, padding, dimension_numbers)
        bias_shape = [out_chan if c == 'C' else 1 for c in out_spec]
        bias_shape = tuple(itertools.dropwhile(lambda x: x == 1, bias_shape))
        k1, k2 = random.split(rng)
        W = W_init(k1, kernel_shape)
        if bias:
            b = b_init(k2, bias_shape)
            return output_shape, (W, b)
        else:
            return output_shape, (W)

    def apply_fun(params, inputs, **kwargs):
        if bias:
            W, b = params
        else:
            W = params
        batchdim = True
        if inputs.ndim == 3:
            batchdim = False
            inputs = np.expand_dims(inputs, 0)
        out = lax.conv_general_dilated(inputs, W, strides, padding, one, one,
                                       dimension_numbers)
        out = out + b if bias else out
        if not batchdim:
            out = out[0]
        return out
    return init_fun, apply_fun


Conv = functools.partial(GeneralConv, ('NHWC', 'HWIO', 'NHWC'))
