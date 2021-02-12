import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax import lax


def _sdeint(f, g, s_init, bm_dim, ts, rng):
  rngs = jax.random.split(rng, len(ts) - 1)

  def f_scan(carry, x):
    t0, t1, rng = x[0], x[1], x[2:].astype('uint32')
    curr_state = carry['curr_state']
    h = t1 - t0
    bm = random.normal(rng, s_init.shape)
    bm_w = bm[:bm_dim]
    f_out = f(t0, curr_state, bm_w)
    g_out = g(t0, curr_state)
    diffusion = jnp.sqrt(h) * bm * g_out
    curr_state = curr_state + h * f_out + diffusion
    carry['curr_state'] = curr_state
    return carry, None

  xs = jnp.concatenate([ts[:-1][..., jnp.newaxis], ts[1:][..., jnp.newaxis], rngs], axis=-1)
  final_state, _ = lax.scan(f_scan, dict(curr_state=s_init), xs)

  return final_state['curr_state']

def sdeint(f, g, s_init, bm_dim, ts, rng):
  """
  f: function drift
  g: function diffusion
  s_init: initial augmented state
  ts: time range / linspace
  prior: whether to sample from prior or posterior

  (x1...xn
  y1...yn
  z1...zn)

  return: list of time states (arrays)
  """
  rngs = jax.random.split(rng, len(ts) - 1)

  def f_scan(carry, x):
    t0, t1, rng = x[0], x[1], x[2:].astype('uint32')
    curr_state = carry['curr_state']
    h = t1 - t0
    bm = random.normal(rng, s_init.shape)
    bm_w = bm[:bm_dim]
    f_out = f(t0, curr_state, bm_w)
    g_out = g(t0, curr_state)
    diffusion = jnp.sqrt(h) * bm * g_out
    curr_state = curr_state + h * f_out + diffusion
    carry['curr_state'] = curr_state
    return carry, curr_state
  xs = jnp.concatenate([ts[:-1][..., jnp.newaxis], ts[1:][..., jnp.newaxis], rngs], axis=-1)
  _, states = lax.scan(f_scan, dict(curr_state=s_init), xs)

  return states
