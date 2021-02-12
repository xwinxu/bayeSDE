import collections
import jax
import jax.numpy as jnp
from jax.experimental import stax
from jax.nn.initializers import zeros

import brax._impl.diffeq_layers as dq


class Layer(collections.namedtuple('Layer', 'init, apply')):
    """Layer consists of a pair of functions: (init, apply)."""


def Swish_(out_dim, beta_init=0.5):
  """Trainable Swish function.
  """
  swish = lambda beta, x: jnp.array(x) * jax.nn.sigmoid(jnp.array(x) * jax.nn.softplus(beta)) # no / 1.1 for lipschitz

  def init_fun(rng, input_shape):
    output_shape = input_shape[:-1] + (out_dim,)
    beta0 = jnp.ones((out_dim,)) * beta_init
    return input_shape, (jnp.array(beta0),)

  def apply_fun(params, inputs, **kwargs):
    beta_, = params
    ret = swish(beta_, inputs)
    return ret
  
  return init_fun, apply_fun

Rbf = stax.elementwise(lambda x: jnp.exp(-x ** 2))
ACTFNS = {"softplus": stax.Softplus, "tanh": stax.Tanh, "elu": stax.Elu, "rbf": Rbf, "swish": Swish_}

def MLP(hidden_dims=[1, 64, 1], actfn="softplus", xt=False, ou_dw=False, p_scale=-1.0, nonzero_w=-1.0, nonzero_b=-1.0):

    def make_layer(input_shape):
        _actfn = ACTFNS[actfn]
        layers = []
        for d_out in hidden_dims:
            if actfn == "swish":
                _actfn = _actfn(d_out)
            if xt:
                layers.append(dq.ConcatSquashLinear(d_out))
                layers.append(dq.DiffEqWrapper(_actfn))
            else:
                layers.append(stax.Dense(d_out))
                layers.append(_actfn)
            if actfn == "swish": # reset for next output shape
                _actfn = ACTFNS[actfn]
        # Zero init last layer unless otherwise specified.
        W_init = b_init = zeros
        if nonzero_w != -1.0:
            print(f"he_normal w init scaled by {nonzero_w}")
            W_init = he_normal(scale_by=nonzero_w)
        if nonzero_b != -1.0:
            print(f"normal b init scaled by {nonzero_b}")
            b_init = normal(scale_by=nonzero_b)
        if xt: # time dependent
            layers.append(dq.ConcatSquashLinear(input_shape[-1], W_init=W_init, b_init=b_init))
        else:
            layers.append(stax.Dense(input_shape[-1], W_init=W_init, b_init=b_init))
        # scale output of final drift linear layer to balance out with diffusion
        if p_scale != -1.0 and xt: # this only works for our time dependent layers for now...
            print(f"drift layer output scaled by exp({p_scale})")
            exp_scale = Exp(input_shape[-1], p_init=p_scale)
            layers.append(dq.DiffEqWrapper(exp_scale))
        return stax.serial(*layers)

    _layer = dq.shape_dependent(make_layer) if xt else stax.shape_dependent(make_layer)
    if ou_dw:
        ou_prior = dq.DiffEqWrapper(Affine(-1, 0.))
        _layer = Additive(ou_prior, _layer)
    layer = Layer(*_layer)
    return layer

def Affine(mult=0., const=1e-3):
    def init_fun(rng, input_shape):
        return input_shape, ()

    def apply_fun(params, inputs, **kwargs):
        return inputs * mult + const

    return init_fun, apply_fun

def Additive(layer1, layer2):
    """
    Learns the difference of two layers.
    layer1: prior drift
    layer2: posterior drift layer
    """
    init_fn1, apply_fn1 = layer1
    init_fn2, apply_fn2 = layer2

    def init_fun(rng, input_shape):
        output1, params1 = init_fn1(rng, input_shape)
        output2, params2 = init_fn2(rng, input_shape)
        return output1, (params1, params2)

    def apply_fun(params, inputs, **kwargs):
        params1, params2 = params
        t1, out1 = apply_fn1(params1, inputs, **kwargs)
        t2, out2 = apply_fn2(params2, inputs, **kwargs)
        return t2, out2 + out1

    return init_fun, apply_fun

def _augment(x, pad):
    """Augment inputs by padding zeros to final dimension.
    e.g. input dimension in 1D, number of channels in 2D conv
    """
    z = jnp.zeros(x.shape[:-1] + (1 * pad,))
    return jnp.concatenate([x, z], axis=-1)


def Augment(pad_zeros):
    """Wrapper for augmented neural ODE.
    """

    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (pad_zeros + input_shape[-1],)
        return output_shape, ()

    def apply_fun(params, inputs, **kwargs):
        x = inputs
        xzeros = _augment(x, pad_zeros)
        return xzeros

    return Layer(init_fun, apply_fun)


def _squeeze(input, downscale_factor=2):
    '''
    [:, C, H*r, W*r] -> [:, C*r^2, H, W]
    '''

    # NHWC -> NCHW
    input = input.transpose((0, 3, 1, 2))

    batch_size, in_channels, in_height, in_width = input.shape
    out_channels = in_channels * (downscale_factor**2)

    out_height = in_height // downscale_factor
    out_width = in_width // downscale_factor

    input_view = input.reshape(batch_size, in_channels, out_height, downscale_factor, out_width, downscale_factor)

    output = input_view.transpose((0, 1, 3, 5, 2, 4))
    output = output.reshape(batch_size, out_channels, out_height, out_width)

    # NCHW -> NHWC
    output = output.transpose((0, 2, 3, 1))

    return output


def SqueezeDownsample(pad_zeros):

    def init_fun(rng, input_shape):
        n, h, w, c = input_shape
        assert h % 2 == 0
        assert w % 2 == 0
        output_shape = (n, h // 2, w // 2, c * 4)
        return output_shape, ()

    def apply_fun(params, inputs, **kwargs):
        return _squeeze(inputs)

    return Layer(init_fun, apply_fun)


def build_fx(block_type, input_size, fx_dim, fx_actfn):
    actfn = ACTFNS[fx_actfn]
    if fx_actfn == "swish":
        actfn = actfn(fx_dim)

    if block_type == 0:
        fx = Layer(
            *stax.serial(
                dq.ConcatConv2D(fx_dim, (3, 3), padding="SAME"),
                dq.DiffEqWrapper(actfn),
                dq.ConcatConv2D(fx_dim, (3, 3), stride=2, padding="SAME"),
                dq.DiffEqWrapper(actfn),
                dq.ConcatConv2D(fx_dim, (3, 3), stride=2, padding="SAME", transpose=True),
                dq.DiffEqWrapper(actfn),
                # NOTE: input_size[0] if NCHW
                dq.ConcatConv2D(input_size[-1], (3, 3), padding="SAME"),
            )
        )
    elif block_type == 1:
        fx = Layer(
            *stax.serial(
                dq.ConcatConv2D(fx_dim, (3, 3), padding="SAME"),
                dq.DiffEqWrapper(actfn),
                dq.ConcatConv2D(fx_dim, (1, 1), padding="SAME"),
                dq.DiffEqWrapper(actfn),
                dq.ConcatConv2D(input_size[-1], (3, 3), padding="SAME"),
            )
        )
    elif block_type == 2:
        fx = Layer(
            *stax.serial(
                dq.ConcatConv2D(fx_dim, (3, 3), padding="SAME"),
                dq.DiffEqWrapper(actfn),
                dq.ConcatConv2D(input_size[-1], (3, 3), padding="SAME"),
            )
        )
    else:
        raise ValueError(f"Invalid block_type {block_type}")

    return fx
