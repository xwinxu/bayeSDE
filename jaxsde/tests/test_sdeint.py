from jax import random
import jax.numpy as np
from jax.config import config
config.update("jax_enable_x64", True)

from jaxsde import make_brownian_motion
from jaxsde import time_reflect_ito, time_reflect_stratonovich, ito_to_stratonovich,\
    stratonovich_to_ito, ito_integrate, stratonovich_integrate

rng = random.PRNGKey(0)
delta_t = 0.001


def make_example_sde(dt=0.1*2**-8):
    D = 3
    ts = np.array([0.1, 0.2])
    y0 = np.linspace(0.1, 0.9, D)

    def f(y, t, args):
        return -np.sqrt(t) - y + 0.1 - np.mean((y + 0.2)**2)

    def g(y, t, args):
        return y**2 + np.sin(0.1 * y) + np.cos(t)

    b = make_brownian_motion(ts[0], np.zeros(y0.shape), ts[1], rng)

    return f, g, b, y0, ts, dt


def test_ito_milstein_vs_euler():
    # Check that for a small enough step size, Euler matches Milstein
    f, g, b, y0, ts, dt = make_example_sde(dt=0.1 * 2**-8)

    ys  = ito_integrate(   f, g, y0, ts, b, dt)
    ys2 = ito_integrate(f, g, y0, ts, b, dt, method='euler_maruyama')
    assert np.allclose(ys, ys2, rtol=1e-02, atol=1e-02)


def test_integrating_noise_noop():
    # Check that integrating changes in Brownian motion gives the same endpoint as the original noise.
    f, g, b, y0, ts, dt = make_example_sde()

    # Create an SDE that will only integrate the noise.
    y0 = y0*0.0
    f = lambda y, t, args: np.zeros(y0.shape)
    g = lambda y, t, args: np.ones(y0.shape)

    sols_i = ito_integrate(f, g, y0, ts, b, dt)
    sols_s = stratonovich_integrate(f, g, y0, ts, b, dt)

    bs = np.array([b(ti) for ti in ts])
    assert np.allclose(bs, sols_i, rtol=1e-03, atol=1e-03)
    assert np.allclose(bs, sols_s, rtol=1e-03, atol=1e-03)


def test_ito_to_strat_int():
    # Run a system forwards then backwards,
    # and check that we end up in the same place.
    f, g, b, y0, ts, dt = make_example_sde()
    fs, gs = ito_to_stratonovich(f, g)

    iys = ito_integrate(         f,  g,  y0, ts, b, dt)
    sys = stratonovich_integrate(fs, gs, y0, ts, b, dt)
    assert np.allclose(iys[-1], sys[-1], rtol=1e-02, atol=1e-02)


def test_strat_to_ito_int():
    # Run a system forwards then backwards,
    # and check that we end up in the same place.
    f, g, b, y0, ts, dt = make_example_sde()
    fi, gi = stratonovich_to_ito(f, g)

    iys = ito_integrate(        fi, gi, y0, ts, b, dt)
    sys = stratonovich_integrate(f,  g, y0, ts, b, dt)
    assert np.allclose(iys[-1], sys[-1], rtol=1e-02, atol=1e-02)


def test_back_and_forth_strat():
    # Run a system forwards then backwards,
    # and check that we end up in the same place.
    f, g, b, y0, ts, dt = make_example_sde()
    fr, gr, br, tr = time_reflect_stratonovich(f, g, b, ts)

    ys  = stratonovich_integrate(f,  g,  y0,     ts, b,  dt)
    rys = stratonovich_integrate(fr, gr, ys[-1], tr, br, dt)[::-1]
    assert np.allclose(ys, rys, rtol=5e-03, atol=5e-03)


def test_back_and_forth_ito():
    # Run a system forwards then backwards,
    # and check that we end up in the same place.
    f, g, b, y0, ts, dt = make_example_sde(dt=0.0001)
    fr, gr, br, tr = time_reflect_ito(f, g, b, ts)

    ys  = ito_integrate(f,  g,  y0,     ts, b,  dt)
    rys = ito_integrate(fr, gr, ys[-1], tr, br, dt)[::-1]
    assert np.allclose(ys[0], rys[0], rtol=1e-03, atol=1e-03)


test_back_and_forth_strat()
test_ito_milstein_vs_euler()
test_integrating_noise_noop()
test_ito_to_strat_int()
test_strat_to_ito_int()
test_back_and_forth_ito()
