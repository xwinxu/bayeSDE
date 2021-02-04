from jax import random
import jax.numpy as np

from jaxsde import make_brownian_motion
from jaxsde import time_reflect_ito, time_reflect_stratonovich, \
                   ito_to_stratonovich, stratonovich_to_ito

D = 10
rng = random.PRNGKey(0)
delta_t = 0.0001

def make_example_sde():
    t0 = 0.1
    t1 = 2.2
    y0 = np.linspace(0.1, 0.9, D)
    args = np.linspace(0.1, 0.4, 4)

    def f(y, t, args):
        return -np.sqrt(t) - y + 0.1 - args[0] * np.mean((y + 0.2)**2) + args[1]
    def g(y, t, args):
        return args[2] * y**2 + np.sin(0.1 * y) + np.cos(t) + args[3]

    ts = np.array([t0, t1])
    return f, g, y0, ts, args

def test_double_reflect_stratonovich():
    # Check that reflecting twice gives the same answer.
    f, g, ts, y0, args = make_example_sde()
    b = make_brownian_motion(ts[0], np.zeros(y0.shape), ts[-1], rng)
    f2, g2, b2, t2 = time_reflect_stratonovich(*time_reflect_stratonovich(f, g, b, ts))
    t = 0.1

    assert(np.all(ts == t2))
    assert(np.all(f(y0, t, args) == f2(y0, t, args)))
    assert(np.all(g(y0, t, args) == g2(y0, t, args)))
    assert(np.all(b(t) == b2(t)))

def test_ito_to_strat_and_back():
    # Check that ito_to_stratonovich(stratonovich_to_ito) is the identity.
    f, g, ts, y0, args = make_example_sde()
    f2, g2 = ito_to_stratonovich(*stratonovich_to_ito(f, g))
    t = 0.1

    assert(np.all(f(y0, t, args) == f2(y0, t, args)))
    assert(np.all(g(y0, t, args) == g2(y0, t, args)))


def test_double_reflect_ito():
    # Check that reflecting twice gives the same answer.
    f, g, ts, y0, args = make_example_sde()
    b = make_brownian_motion(ts[0], np.zeros(y0.shape), ts[-1], rng)
    f2, g2, b2, t2 = time_reflect_ito(*time_reflect_ito(f, g, b, ts))
    t = 0.1

    assert(np.all(ts == t2))
    assert(np.allclose(f(y0, t, args), f2(y0, t, args)))
    assert(np.allclose(g(y0, t, args), g2(y0, t, args)))
    assert(np.all(b(t) == b2(t)))


def test_reflect_ito_two_ways():
    # Check that reflect_ito = strat_to_ito( reflect_strat ( ito_to_strat )))
    f, g, ts, y0, args = make_example_sde()
    b = make_brownian_motion(ts[0], np.zeros(y0.shape), ts[-1], rng)

    fr, gr, br, tr = time_reflect_ito(f, g, b, ts)

    fi, gi = ito_to_stratonovich(f, g)
    f2, g2, b3, t3 = time_reflect_stratonovich(fi, gi, b, ts)
    f3, g3 = stratonovich_to_ito(f2, g2)
    t = 0.1

    assert(np.all(tr == t3))
    assert(np.allclose(fr(y0, -t, args), f3(y0, -t, args)))
    assert(np.allclose(gr(y0, -t, args), g3(y0, -t, args)))
    assert(np.all(br(-t) == b3(-t)))


# def test_diagonal_gdg():
#     # Check that the fast formula is correct for diagonal functions.
#     f, g, ts, y0, args = make_example_sde()
