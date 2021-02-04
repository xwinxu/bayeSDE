import numpy as onp

from jax import random
import jax.numpy as np
from jax.config import config
from jaxsde import make_brownian_motion
config.update("jax_enable_x64", True)


D = 2
rng = random.PRNGKey(0)
delta_t = 0.0001


def test_mean_and_var():
    t0 = 0.0
    t1 = 3.0
    y0 = np.linspace(0.1, 0.9, D)
    num_samples = 300

    vals = onp.zeros((num_samples, D))
    for i in range(num_samples):
        rng = random.PRNGKey(i)
        bm = make_brownian_motion(t0, np.zeros(y0.shape), t1, rng)
        vals[i, :] = bm(t1)

    assert np.allclose(np.mean(vals), 0.0, atol=1e-1, rtol=1e-1)
    assert np.allclose(np.var(vals),  t1,  atol=1e-1, rtol=1e-1)


def test_mean_and_var_mid():
    t0 = 0.0
    t1 = 3.0
    y0 = np.linspace(0.1, 0.9, D)
    num_samples = 500

    vals = onp.zeros((num_samples, D))
    for i in range(num_samples):
        rng=random.PRNGKey(i)
        bm = make_brownian_motion(t0, np.zeros(y0.shape), t1, rng)
        vals[i, :] = bm(t1/2.0)

    print(np.mean(vals), np.var(vals))
    assert np.allclose(np.mean(vals), 0.0, atol=1e-1, rtol=1e-1)
    assert np.allclose(np.var(vals), t1/2.0, atol=1e-1, rtol=1e-1)

# TODO: test indepdendence of dimensions
