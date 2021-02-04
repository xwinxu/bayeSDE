"""Virtual Brownian tree.
"""
from functools import partial

from jax import jit, random
from jax.lax import scan
import jax.numpy as np
from jax.ops import index, index_update


def scaled_brownian_motion(original_bm, t0, t1):
    # Turns a standard Brownian motion on [0,1] into one on [t0, t1].
    # cf. section "Wiener representation" of https://en.wikipedia.org/wiki/Wiener_process
    # Initial time `t0` should be at most the end time `t1`.
    # The Brownian motion to be scaled `original_bm` should be a function.'
    return lambda t: np.sqrt(t1 - t0) * original_bm((t - t0) / (t1 - t0))


def brownian_bridge_point(t0, x0, t1, x1, t, rng, rep=0):
    mean = (x0 * (t1 - t) + x1 * (t - t0)) / (t1 - t0)
    std = np.sqrt((t1 - t) * (t - t0) / (t1 - t0))
    z = sample_with_rep(rng, x0.shape, rep)
    return mean + std * z


@partial(jit, static_argnums=(5, 7))
def virtual_brownian_tree(ot0, ox0, ot1, ox1, t, depth, orng, rep):
    # This algorithm samples from a standard Brownian bridge with O(1) memory
    # and O(-log (1/eps)) time.  From https://arxiv.org/abs/2001.01328

    def scan_fun(carry, _):
        t0, x0, t1, x1, rng = carry
        t_mid = (t0 + t1) / 2.0
        x_mid = brownian_bridge_point(t0, x0, t1, x1, t_mid, rng, rep)
        left_rng, right_rng = random.split(rng)

        b = t < t_mid
        xl = np.where(b, x0, x_mid)
        tl = np.where(b, t0, t_mid)
        xr = np.where(b, x_mid, x1)
        tr = np.where(b, t_mid, t1)
        r = np.where(b, left_rng, right_rng)

        new_carry = tl, xl, tr, xr, r

        return new_carry, ()

    init_carry = ot0, ox0, ot1, ox1, orng
    out_carry, _ = scan(scan_fun, init_carry, None, depth)
    ft0, fx0, ft1, fx1, frng = out_carry
    return brownian_bridge_point(ft0, fx0, ft1, fx1, t, frng, rep)


# @partial(jit, static_argnums=(1, 2))
# def sample_with_rep(rng, shape, rep):
#     if rep == 0:
#         return random.normal(rng, shape)

#     free_shape = (shape[0] - 2 * rep, *shape[1:])
#     rep_shape = (rep, *shape[1:])

#     rngs = random.split(rng)
#     free_z = random.normal(rngs[0], free_shape)
#     rep_z = random.normal(rngs[1], rep_shape)
#     z = np.concatenate([free_z, rep_z, rep_z], axis=0)
#     return z


@partial(jit, static_argnums=(1, 2))
def sample_with_rep(rng, shape, rep):
    n = shape[0]
    z = random.normal(rng, shape)
    z = index_update(z, index[n - rep:], z[n - rep * 2:n - rep])
    return z


def make_standard_brownian_motion(x0, depth, rng, rep):
    # Curry standard Brownian motion sampler.

    # Sample value at t1.
    end_rng, path_rng = random.split(rng)
    w_end = sample_with_rep(end_rng, x0.shape, rep)

    def b(t):
        # TODO: Return nans outside of valid range.
        # if t < 0.0: raise("Error: Brownian bridge queried before start time")
        # if t > 1.0: raise("Error: Brownian bridge queried after end time")
        return virtual_brownian_tree(0.0, np.zeros(x0.shape), 1.0, w_end, t, depth, path_rng, rep)
    return b


def make_brownian_motion(t0, x0, t1, rng, depth=10, rep=0):
    # Curry brownian motion sampler from time t0 to t1.
    standard_bm = make_standard_brownian_motion(x0, depth, rng, rep)
    return scaled_brownian_motion(standard_bm, t0, t1)


if __name__ == "__main__":

    t0 = np.array(0.0)
    t1 = np.array(1.0)

    rng = random.PRNGKey(0)

    b = make_brownian_motion(t0, np.zeros(10), t1, rng, rep=2)

    print(b(0.0))
    print(b(1.0))
    print(b(0.1))
