import numpy as onp

import jax
import jax.numpy as np
from jax.flatten_util import ravel_pytree

from .sdeint import ito_integrate
from .sde_utils import make_gdg_prod
from .sde_vjp import vjp_ito_integrate


# `jax.defvjp` doesn't accept high-order functions, so create wrapper s.t. we may call `jax.grad` on `ito_integrate`.
# TODO(lxuechen): This is not functionally pure! `bm` should just be a function. Change `bm.renew()` to `renew(bm)`.
def make_ito_integrate(flat_f, flat_g, ts, dt, bm, method='milstein'):
    # `flat_f` and `flat_g` each takes in *flat* states and parameters and returns *flat* gradients.

    # Make fast jitted helper functions for Milstein correction.
    @jax.jit
    def flat_g_prod(flat_y, t, args, noise):
        return flat_g(flat_y, t, args) * noise
    flat_gdg = jax.jit(make_gdg_prod(flat_g_prod))

    @jax.custom_transforms
    def ito_integrate_(flat_y0, flat_args):
        return ito_integrate(
            flat_f, flat_g,
            flat_y0,
            ts, bm, dt,
            args=flat_args,
            g_prod=flat_g_prod,
            gdg=flat_gdg,
            method=method)  # (T, batch_size * D)

    def vjp_all(flat_y0, flat_args):
        ans = ito_integrate(
            flat_f, flat_g,
            flat_y0,
            ts, bm, dt,
            args=flat_args,
            g_prod=flat_g_prod,
            gdg=flat_gdg,
            method=method)  # (T, batch_size * D).

        def actual_vjp_all(cotan):
            T, _ = cotan.shape

            v_flat_y, v_flat_args = cotan[-1, :], np.zeros_like(flat_args)
            for i in range(T - 1, 0, -1):
                ts_local = np.array([ts[i - 1], ts[i]])
                _, (v_flat_y, v_flat_args) = vjp_ito_integrate(
                    v_yt=v_flat_y,
                    v_argst=v_flat_args,
                    yt=ans[i, :],
                    f=flat_f, g=flat_g,
                    ts=ts_local, bm=bm, dt=dt,
                    args=flat_args, method=method)
                v_flat_y = v_flat_y + cotan[i - 1, :]
            return v_flat_y, v_flat_args

        return ans, actual_vjp_all
    jax.defvjp_all(ito_integrate_, vjp_all)

    return ito_integrate_


def make_sde(unpack_args):

    def flat_f(flat_y, t, flat_args):
        flat_args_f, _ = unpack_args(flat_args)
        return np.sin(flat_y) + np.cos(t) + flat_args_f * 0.1

    def flat_g(flat_y, t, flat_args):
        del t
        _, flat_args_g = unpack_args(flat_args)
        return np.sin(flat_y) + flat_args_g * 0.3

    return flat_f, flat_g


def numerical_gradient(f, x, eps=1e-4):
    flat_x, unravel = ravel_pytree(x)
    D = len(flat_x)
    g = onp.zeros_like(flat_x)
    for i in range(D):
        d = onp.zeros_like(flat_x)
        d[i] = eps
        g[i] = (f(unravel(flat_x + d)) - f(unravel(flat_x - d))) / (2.0 * eps)
    return g


def test_make_ito_integrate(D=2):
    key = jax.random.PRNGKey(0)
    fk, gk = jax.random.split(key, num=2)
    f_flat_args = jax.random.normal(key=fk, shape=(D,))
    g_flat_args = jax.random.normal(key=gk, shape=(D,))
    args = (f_flat_args, g_flat_args)
    flat_args, unpack_args = jax.flatten_util.ravel_pytree(args)
    flat_f, flat_g = make_sde(unpack_args)
    bm = lambda t: t

    # TODO: `ts` and `dt` should *not* be part of this signature! But jax automatically traces all input args.
    ts = np.array([0., .15, .3])
    dt = 1e-4
    ito_integrate_ = make_ito_integrate(flat_f=flat_f, flat_g=flat_g, ts=ts, dt=dt, bm=bm)

    # Define loss and gradient function.
    def loss_fn(flat_y0, flat_args):
        yt = ito_integrate_(flat_y0, flat_args)
        return np.mean((yt - 3.) ** 2.)
    grad_fn = jax.grad(loss_fn, argnums=(0, 1))

    # Actual computation.
    flat_y0 = jax.random.normal(key=key, shape=(D,))
    dy, dargs = grad_fn(flat_y0, flat_args)
    print("vjp grads      :", dy, dargs)

    # Numerical grad.
    def loss_fn_wrt_flat_y0(flat_y0):
        return loss_fn(flat_y0, flat_args)

    def loss_fn_wrt_flat_args(flat_args):
        return loss_fn(flat_y0, flat_args)

    dy_numerical = numerical_gradient(loss_fn_wrt_flat_y0, flat_y0)
    dargs_numerical = numerical_gradient(loss_fn_wrt_flat_args, flat_args)
    print("numerical grads:", dy_numerical, dargs_numerical)

    onp.testing.assert_allclose(dy, dy_numerical, atol=1e-2, rtol=1e-2)
    onp.testing.assert_allclose(dargs, dargs_numerical, atol=1e-2, rtol=1e-2)


if __name__ == '__main__':
    test_make_ito_integrate()
