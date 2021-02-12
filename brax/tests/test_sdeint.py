"""
Check SDE solver for a noisy Lorenz system.
"""
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
from brax.utils.sdeint import sdeint


def f_lorenz(t, s):
  """
  Drift.
  t: time
  s: array[state]
  """
  x, w, kl = s[0], s[1], s[2]
  dx = w - x
  dw = -1 * x * kl - w
  dkl = x * w - 2
  f_out = jnp.stack([dx, dw, dkl])
  return f_out

def g_lorenz(t, s):
  """
  Diffusion.
  t: time
  s: array[states]
  """
  return jnp.ones_like(s) * 0.01

def plot(states):
  plt.figure(figsize=(12, 8), facecolor='white')
  plt.subplot(131, frameon=False)
  plt.plot(states[:, 0], states[:, 1])
  plt.subplot(132, frameon=False)
  plt.plot(states[:, 0], states[:, 2])
  plt.subplot(133, frameon=False)
  plt.plot(states[:, 1], states[:, 2])

  plt.show()


def test_sdeint():
  rng = random.PRNGKey(0)
  s_init = jnp.ones(3) * 0.1
  ts = jnp.linspace(0, 100, 1000)

  states = sdeint(f_lorenz, g_lorenz, s_init, ts, rng)

  states = jnp.stack(states)  # 1000 x 3
  print(states.shape)

  plot(states)

if __name__ == "__main__":
  test_sdeint()
