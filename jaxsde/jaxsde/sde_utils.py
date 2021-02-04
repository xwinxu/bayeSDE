import jax.numpy as np
from jax import vjp, jvp

def time_reflect_stratonovich(f, g, b, ts):
    rf = lambda y, t, args: -f(y, -t, args)
    rg = lambda y, t, args:  g(y, -t, args)
    rb = lambda    t: b(-t)
    return rf, rg, rb, -ts[::-1]


def ito_to_stratonovich(f, g):
    gdg = make_gdg(g)
    corrected_drift = lambda y, t, args: f(y, t, args) - 0.5 * gdg(y, t, args)
    return corrected_drift, g


def stratonovich_to_ito(f, g):
    gdg = make_gdg(g)
    corrected_drift = lambda y, t, args: f(y, t, args) + 0.5 * gdg(y, t, args)
    return corrected_drift, g


def time_reflect_ito(f, g, b, ts, gdg=None):
    if gdg is None:
        gdg = make_gdg(g)
    rf = lambda y, t, args: -f(y, -t, args) + gdg(y, -t, args)
    rg = lambda y, t, args: g(y, -t, args)
    rb = lambda    t: b(-t)
    return rf, rg, rb, -ts[::-1]


def time_reflect_ito_prod(f, g_prod, b, ts, gdg_prod=None):
    if gdg_prod is None:
        gdg_prod = make_gdg_prod(g_prod)
    rf = lambda y, t, args: -f(y, -t, args) + gdg_prod(y, -t, args)
    rg = lambda y, t, args, noise: g_prod(y, -t, args, noise)
    rb = lambda    t: b(-t)
    return rf, rg, rb, -ts[::-1]


def make_gdg(g):
    # Only valid for elementwise functions.
    def gdg(y, t, args):
        g_y_only = lambda y: g(y, t, args)
        return jvp(g_y_only, (y,), (g_y_only(y),))[1]
    return gdg


def diag_jac(g, y, t, args):
    # Only valid for elementwise functions.
    g_y_only = lambda y: g(y, t, args)
    return jvp(g_y_only, (y,), (np.ones(y.shape),))[1]


def make_gdg_prod(g_prod):
    # Only valid for elementwise functions.
    g = lambda y, t, args: g_prod(y, t, args, np.ones(np.size(y)))
    def gdg_prod(y, t, args, v):
        g_y_only = lambda y: g(y, t, args)
        return jvp(g_y_only, (y,), (v * g_y_only(y),))[1]
    return gdg_prod


def check_symmetric_jac(g, y):
    val1, fwd = jvp(g, (y,), (np.ones(y.shape),))
    val2, vjpf = vjp(g, y)
    rev = vjpf(np.ones(y.shape))
    assert np.allclose(val1, val2)
    assert np.allclose(fwd, rev)
