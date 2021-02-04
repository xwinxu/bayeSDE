import matplotlib.pyplot as plt

from jax import vmap, random, jit
import jax.numpy as np
from jax.config import config
config.update("jax_enable_x64", True)

from jaxsde.sdeint import sdeint_ito



def plot_gradient_field(ax, func, xlimits, ylimits, numticks=30):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    zs = vmap(func)(Y.ravel(), X.ravel())
    Z = zs.reshape(X.shape)
    ax.quiver(X, Y, np.ones(Z.shape), Z)
    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)


def plot_sde_soln():
    # Shows a few different samples from the same SDE.

    def f(y, t, args):
        arg1, arg2 = args
        return -y - np.sin(t) - np.cos(t) * arg1 + arg2

    def g(y, t, args):
        return 1.0

    t0 = 0.1
    t1 = 4.9
    nsteps = 200
    nsamples = 10
    ts = np.linspace(t0, t1, nsteps)
    y0 = np.array([1.])
    fargs = (1.0, 0.0)

    def int(rng):
        return sdeint_ito(f, g, y0, ts, rng, fargs, dt=0.001)

    ints = jit(vmap(int))

    rng = random.PRNGKey(0)
    rngs = random.split(rng, nsamples)
    ys = ints(rngs)
    ys = ys.reshape((nsamples, nsteps)).T

    # Set up figure.
    fig = plt.figure(figsize=(8, 6), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.ion()
    plt.show(block=False)
    plt.cla()
    f_no_args = lambda y, t: f(y, t, fargs)
    plot_gradient_field(ax, f_no_args, xlimits=[t0, t1], ylimits=[-2.1, 2.1])
    ax.plot(ts, ys)
    ax.set_xlabel('t')
    ax.set_ylabel('y')
    plt.draw()
    plt.pause(100)


if __name__ == "__main__":
    plot_sde_soln()
