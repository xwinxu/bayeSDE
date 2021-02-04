import matplotlib.pyplot as plt

from jax import jit, grad
from jax import vmap, random
import jax.numpy as np
from jaxsde import sdeint_ito
from jaxsde.svi import mc_elbo
from jax.config import config
from jax.experimental import optimizers
config.update("jax_enable_x64", True)


def plot_gradient_field(ax, func, xlimits, ylimits, numticks=30):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    zs = vmap(func)(Y.ravel(), X.ravel())
    Z = zs.reshape(X.shape)
    ax.quiver(X, Y, np.ones(Z.shape), Z)
    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)


def toy_svi():
    D = 1
    z0 = np.zeros(D)
    t0 = 0.
    t1 = 1.
    nsteps = 1000
    ts = np.linspace(t0, t1, nsteps)
    nsamples = 4

    prior_drift = lambda z, t, args: -z
    prior_diffusion = lambda z, t, args: np.ones_like(z)
    prior_params = ()

    posterior_drift = lambda z, t, args: -z * args[0] + t * args[1]
    init_posterior_params = np.array([-0.3, -0.2])

    log_likelihood = lambda z, params: np.sum(-z**2)
    log_likelihood_params = ()

    def elbo(posterior_params, rng):
        return mc_elbo(z0, t0, t1, prior_params, posterior_params, log_likelihood_params,
                prior_drift, prior_diffusion, posterior_drift, log_likelihood, rng)
    velbo = vmap(elbo, in_axes=(None, 0))

    def batched_loss(posterior_params, t, num_samples=10):
        rngs = random.split(random.PRNGKey(t), num_samples)
        return -np.mean(velbo(posterior_params, rngs))
        #return -np.mean(elbo(rngs[0]))  # MATT: This alternate version wihtout vmap works

    print(batched_loss(init_posterior_params, 0))
    print(grad(batched_loss)(init_posterior_params, 0))

    # Set up figure.
    fig = plt.figure(figsize=(8, 6), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.ion()
    plt.show(block=False)

    def sde_sample(rng, posterior_params):
        return sdeint_ito(posterior_drift, prior_diffusion, z0, ts, rng, posterior_params, dt=0.001)

    sde_samples = jit(vmap(sde_sample, in_axes=(0, None)))
    rngs = random.split(random.PRNGKey(0), nsamples)

    def callback(posterior_params, t):

        print(batched_loss(posterior_params, t))

        ys = sde_samples(rngs, posterior_params)
        ys = ys.reshape((nsamples, nsteps)).T

        plt.cla()
        ax.cla()
        f_no_args = lambda y, t: posterior_drift(y, t, posterior_params)
        plot_gradient_field(ax, f_no_args, xlimits=[t0, t1], ylimits=[-1.1, 4.1])
        ax.plot(ts, ys)
        ax.set_xlabel('t')
        ax.set_ylabel('y')
        plt.draw()
        plt.pause(0.01)

    opt_init, opt_update, get_params = optimizers.adam(step_size=0.1)
    opt_state = opt_init(init_posterior_params)

    @jit
    def update(i, opt_state):
        params = get_params(opt_state)
        gradient = grad(batched_loss)(params, i)
        return opt_update(i, gradient, opt_state)

    # Main loop.
    print("Optimizing...")
    for t in range(1000):
        opt_state = update(t, opt_state)
        params = get_params(opt_state)
        if t % 50 == 1:
            callback(params, t)
    plt.show(block=True)


if __name__ == "__main__":
    toy_svi()


