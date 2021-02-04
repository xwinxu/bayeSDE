import numpy as onp

import jax.numpy as np
from jax import random, jacobian, grad, jit
from jax.flatten_util import ravel_pytree
from jax.config import config
config.update("jax_enable_x64", True)
from jax import test_util as jtu

from jaxsde import ito_integrate, vjp_ito_integrate, stratonovich_integrate, sdeint_ito
from jaxsde.sde_vjp import vjp_strat_integrate, make_ito_adjoint_dynamics
from jaxsde import make_brownian_motion
from jaxsde import make_explicit_sigma, make_explicit_milstein


def make_sde():
    D = 3
    ts = np.array([0.1, 0.5])
    y0 = np.linspace(0.1, 0.9, D)
    args = (np.linspace(0.1, 0.2, 2), np.linspace(0.8, 0.9, 3))

    def f(y, t, args):
        f_args, g_args = args
        return -np.sqrt(t) - y + 0.1 - f_args[0] * np.mean((y + 0.2)**2) + f_args[1]

    def g(y, t, args):
        f_args, g_args = args
        return g_args[0] * (y**2 + np.sin(0.1 * y)) + g_args[1] * np.cos(t) + g_args[2]

    return D, ts, y0, args, f, g


def make_flat_sde():
    D = 3
    ts = np.array([0.1, 0.5])
    y0 = np.linspace(0.1, 0.9, D)
    args = np.linspace(0.1, 0.2, 5)

    def f(y, t, args):
        f_args = args[:2]
        return -np.sqrt(t) - y + 0.1 - f_args[0] * np.mean((y + 0.2)**2) + f_args[1]

    def g(y, t, args):
        g_args = args[2:]
        return g_args[0] * (y**2 + np.sin(0.1 * y)) + g_args[1] * np.cos(t) + g_args[2]

    return D, ts, y0, args, f, g


def test_adjoint_g_dynamics():
    # Check that the function that computes the product of the augmented
    # diffusion dynamics against a vector actually does the same thing as
    # computing the diffusion matrix explicitly.
    D, ts, y0, args, f, g = make_sde()

    flat_args, unravel = ravel_pytree(args)
    def flat_f(y, t, flat_args): return f(y, t, unravel(flat_args))
    def flat_g(y, t, flat_args): return g(y, t, unravel(flat_args))

    aug_y, unpack = ravel_pytree((y0, y0, np.zeros(np.size(flat_args))))
    f_aug, g_aug, aug_gdg = make_ito_adjoint_dynamics(flat_f, flat_g, unpack)

    # Test g_aug
    sigma = make_explicit_sigma(flat_g, unpack)
    explicit = sigma(aug_y, ts[0], flat_args)
    implicit = jacobian(g_aug, argnums=3)(aug_y, ts[0], flat_args, np.ones(y0.shape))
    assert np.allclose(explicit, implicit)

    # Test aug_gdg (Milstein correction factor)
    explicit_milstein = make_explicit_milstein(sigma, aug_y, ts[0], flat_args)
    implicit_milstein = jacobian(aug_gdg, argnums=3)(aug_y, ts[0], flat_args, np.ones(y0.shape))
    print(explicit_milstein)
    print(implicit_milstein)
    assert np.allclose(explicit_milstein, implicit_milstein)


def numerical_gradient(f, x, eps=0.0001):
    flat_x, unravel = ravel_pytree(x)
    D = len(flat_x)
    g = onp.zeros_like(flat_x)
    for i in range(D):
        d = onp.zeros_like(flat_x)
        d[i] = eps
        g[i] = (f(unravel(flat_x + d)) - f(unravel(flat_x - d))) / (2.0 * eps)
    return g


def test_ito_int_vjp():
    D, ts, y0, args, f, g = make_sde()
    flat_args, _ = ravel_pytree(args)
    bm = make_brownian_motion(ts[0], np.zeros(y0.shape), ts[1], random.PRNGKey(0))
    dt = 1e-4
    eps = 1e-6
    method = 'milstein'

    def ito_int(argv):
        y0, args = argv
        ys = ito_integrate(f, g, y0, ts, bm, dt, args, method=method)
        return np.sum(ys[1])

    ys = ito_integrate(f, g, y0, ts, bm, dt, args, method=method)
    v_yt = np.ones_like(y0)
    v_argst = np.zeros_like(flat_args)
    y0_rec, exact_grad = vjp_ito_integrate(v_yt, v_argst, ys[-1], f, g, ts, bm, dt, args, method=method)

    numerical_grad = numerical_gradient(ito_int, (y0, args), eps=eps)

    print("states:",   y0, y0_rec)
    assert np.allclose(y0, y0_rec, rtol=1e-2, atol=1e-02)

    flat_grads, unravel = ravel_pytree(exact_grad)
    print("numerical grad: ", unravel(numerical_grad))
    print("    exact grad: ", exact_grad)
    assert np.allclose(numerical_grad, flat_grads, rtol=1e-2, atol=1e-2)


def test_strat_int_vjp():
    D, ts, y0, args, f, g = make_sde()
    flat_args, _ = ravel_pytree(args)
    bm = make_brownian_motion(ts[0], np.zeros(y0.shape), ts[1], random.PRNGKey(0))
    dt = 1e-4
    eps = 1e-6

    def strat_int(argv):
        y0, args = argv
        ys = stratonovich_integrate(f, g, y0, ts, bm, dt, args)
        return np.sum(ys[1])

    ys = stratonovich_integrate(f, g, y0, ts, bm, dt, args)
    v_yt = np.ones_like(y0)
    v_argst = np.zeros_like(flat_args)
    y0_rec, exact_grad = vjp_strat_integrate(v_yt, v_argst, ys[-1], f, g, ts, bm, dt, args)

    numerical_grad = numerical_gradient(strat_int, (y0, args), eps=eps)

    print("states:",   y0, y0_rec)
    assert np.allclose(y0, y0_rec, rtol=1e-2, atol=1e-02)

    flat_grads, unravel = ravel_pytree(exact_grad)
    print("numerical grad: ", unravel(numerical_grad))
    print("    exact grad: ", exact_grad)
    assert np.allclose(numerical_grad, flat_grads, rtol=1e-2, atol=1e-2)



def test_jax_grads():
    D, ts, y0, args, f, g = make_flat_sde()
    dt = 1e-4
    eps = 1e-3

    @jit
    def ito_int(y0, args):
        ys = sdeint_ito(f, g, y0, ts, random.PRNGKey(0), args, dt)
        return np.sum(ys[1])

    jtu.check_grads(ito_int, (y0, args), modes=["rev"], order=1, rtol=eps, atol=eps)


def test_jax_grads_pytree_args():
    D, ts, y0, args, f, g = make_sde()
    flat_args, _ = ravel_pytree(args)
    dt = 1e-4
    eps = 1e-3

    def ito_int(y0, args):
        ys = sdeint_ito(f, g, y0, ts, random.PRNGKey(0), args, dt)
        return np.sum(ys[1])

    print("Eval:")
    print(ito_int(y0, args))

    print("grad:")
    print(grad(ito_int)(y0, args))

    jtu.check_grads(ito_int, (y0, args), modes=["rev"], order=1, rtol=eps, atol=eps)
