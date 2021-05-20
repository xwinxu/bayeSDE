import math

import jax
import jax.numpy as jnp
from jax.experimental import stax
from jax.flatten_util import ravel_pytree
from jax import tree_util
from jax.lax import stop_gradient

from brax._impl.arch import Layer, build_fx
from jaxsde.jaxsde.sdeint import sdeint_ito_fixed_grid, sdeint_ito


def SDEBNN(fx_block_type, fx_dim, fx_actfn, fw, diff_coef=1e-4, name="sdebnn", stl=False, xt=False, nsteps=20, remat=False, w_drift=True, stax_api=False):

    # This controls the number of function evaluations and the step size.
    ts = jnp.linspace(0.0, 1.0, nsteps)

    def make_layer(input_shape):

        fx = build_fx(fx_block_type, input_shape, fx_dim, fx_actfn)

        # Creates the unflatten_w function.
        rng = jax.random.PRNGKey(0)  # temp; not used.
        x_shape, tmp_w = fx.init(rng, input_shape)
        assert input_shape == x_shape, f"fx needs to have the same input and output shapes but got {input_shape} and {x_shape}"
        flat_w, unflatten_w = ravel_pytree(tmp_w)
        w_shape = flat_w.shape
        del tmp_w

        x_dim = np.abs(np.prod(x_shape))
        w_dim = np.abs(np.prod(w_shape))

        def f_aug(y, t, args):
            x = y[:x_dim].reshape(x_shape)
            flat_w = y[x_dim:x_dim + w_dim].reshape(w_shape)
            dx = fx.apply(unflatten_w(flat_w), (x, t))[
                0] if xt else fx.apply(unflatten_w(flat_w), x)
            if w_drift:
                fw_params = args
                dw = fw.apply(fw_params, (flat_w, t))[
                    0] if xt else fw.apply(fw_params, flat_w)
            else:
                dw = jnp.zeros(w_shape)
            # Hardcoded OU Process.
            u = (dw - (-flat_w)) / \
                diff_coef if diff_coef != 0 else jnp.zeros(w_shape)
            dkl = u**2
            return jnp.concatenate([dx.reshape(-1), dw.reshape(-1), dkl.reshape(-1)])

        def g_aug(y, t, args):
            dx = jnp.zeros(x_shape)
            diff_w = jnp.ones(w_shape) * diff_coef

            if w_drift:
                fw_params = tree_util.tree_map(stop_gradient, args)
                drift_w = fw.apply(fw_params, (flat_w, t))[0] if xt else fw.apply(fw_params, flat_w)
            else:
                drift_w = jnp.zeros(w_shape)

            # Hardcoded OU Process.
            u = (drift_w - (-flat_w)) / \
                diff_coef if diff_coef != 0 else jnp.zeros(w_shape)
            dkl = u if stl else jnp.zeros(w_shape)
            return jnp.concatenate([dx.reshape(-1), diff_w.reshape(-1), dkl.reshape(-1)])

        def init_fun(rng, input_shape):
            output_shape, w0 = fx.init(rng, input_shape)
            flat_w0, unflatten_w = ravel_pytree(w0)
            if w_drift:
                output_shape, fw_params = fw.init(rng, flat_w0.shape)
                assert flat_w0.shape == output_shape, "fw needs to have the same input and output shapes"
            else:
                fw_params = ()
            return input_shape, (flat_w0, fw_params)

        def apply_fun(params, inputs, rng, full_output=False, fixed_grid=True, **kwargs):
            flat_w0, fw_params = params
            x = inputs
            y0 = jnp.concatenate(
                [x.reshape(-1), flat_w0.reshape(-1), jnp.zeros(flat_w0.shape).reshape(-1)])
            rep = w_dim if stl else 0  # STL
            if fixed_grid:
                ys = sdeint_ito_fixed_grid(f_aug, g_aug, y0, ts, rng, fw_params, method="euler_maruyama", rep=rep)
            else:
                print("using stochastic adjoint")
                ys = sdeint_ito(f_aug, g_aug, y0, ts, rng, fw_params, method="euler_maruyama", rep=rep)
            y = ys[-1]  # Take last time value.
            x = y[:x_dim].reshape(x_shape)
            # import pdb; pdb.set_trace()
            kl = jnp.sum(y[x_dim + w_dim:])

            # Hack to turn this into a stax.layer API when deterministic.
            if stax_api:
                return x

            if full_output:
                infodict = {
                    name + "_w": ys[:, x_dim:x_dim + w_dim].reshape(-1, *w_shape)
                }
                return x, kl, infodict

            return x, kl

        if remat:
            apply_fun = jax.checkpoint(apply_fun, concrete=True)
        return init_fun, apply_fun

    return Layer(*stax.shape_dependent(make_layer))


def MeanField(layer, prior_std=0.1, disable=False):

    init_fun, apply_fun = layer

    def wrapped_init_fun(rng, input_shape):
        output_shape, params_mean = init_fun(rng, input_shape)
        params_logstd = tree_util.tree_map(
            lambda x: jnp.zeros_like(x) - 4.0, params_mean)
        return output_shape, (params_mean, params_logstd)

    def wrapped_apply_fun(params, input, rng, **kwargs):
        params_mean, params_logstd = params

        flat_mean, unflatten = ravel_pytree(params_mean)
        flat_logstd, _ = ravel_pytree(params_logstd)

        rng, next_rng = jax.random.split(rng)
        if disable:
            flat_params = flat_mean
        else:
            flat_params = jax.random.normal(
                rng, flat_mean.shape) * jnp.exp(flat_logstd) + flat_mean
        params = unflatten(flat_params)

        if disable:
            kl = jnp.zeros_like(flat_params)
        else:
            kl = normal_logprob(flat_params, flat_mean, flat_logstd) - \
                normal_logprob(flat_params, 0., jnp.log(prior_std))
        output = apply_fun(params, input, rng=next_rng, **kwargs)
        return output, jnp.sum(kl)

    return Layer(wrapped_init_fun, wrapped_apply_fun)


def normal_logprob(z, mean, log_std):
    mean = mean + jnp.zeros(1)
    log_std = log_std + jnp.zeros(1)
    # c = jnp.array([math.log(2 * math.pi)])
    c = math.log(2 * math.pi)
    inv_sigma = jnp.exp(-log_std)
    tmp = (z - mean) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * log_std + c)


def bnn_serial(*layers):
    """Combinator for composing layers in serial.

    Args:
      *layers: a sequence of layers, each an (init_fun, apply_fun) pair.

    Returns:
      A new layer, meaning an (init_fun, apply_fun) pair, representing the serial
      composition of the given sequence of layers.
    """
    nlayers = len(layers)
    init_funs, apply_funs = zip(*layers)

    def init_fun(rng, input_shape):
        params = []
        for init_fun in init_funs:
            rng, layer_rng = jax.random.split(rng)
            input_shape, param = init_fun(layer_rng, input_shape)
            params.append(param)
        return input_shape, params

    def apply_fun(params, inputs, **kwargs):
        rng = kwargs.pop('rng', None)
        rngs = jax.random.split(
            rng, nlayers) if rng is not None else (None,) * nlayers
        total_kl = 0
        infodict = {}
        for fun, param, rng in zip(apply_funs, params, rngs):
            output = fun(param, inputs, rng=rng, **kwargs)
            if len(output) == 2:
                inputs, layer_kl = output
            elif len(output) == 3:
                inputs, layer_kl, info = output
                infodict.update(info)
            else:
                raise RuntimeError(
                    f"Expected 2 or 3 outputs but got {len(output)}.")
            total_kl = total_kl + layer_kl
        return inputs, total_kl, infodict
    return Layer(init_fun, apply_fun)
