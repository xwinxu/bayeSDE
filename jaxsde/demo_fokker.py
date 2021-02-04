import matplotlib.pyplot as plt

from jax import jit, grad, custom_jvp
from jax import vmap, random
import jax.numpy as np

from jax.scipy.stats import norm
from jax.experimental.ode import odeint
from jax.scipy.ndimage import map_coordinates

from jax.config import config
config.update("jax_enable_x64", True)


def fokker_planck(f, g, x, t, p):
    # Inputs: f(x, time): drift function of an Ito SDE.
    #         g(x, time): diffusion function of an Ito SDE.
    #         i.e. dx = f(x, t)dt + g(x, t) dW
    # Returns:
    #         The rate of change of density p(x) wrt time.
    #
    # dp(x,t)/dt = - grad_x(f * p) + 0.5 * grad_xx(g^2 p)
    f_times_p         = lambda x, t, p: f(x, t) * p(x)
    g_squared_times_p = lambda x, t, p: 0.5 * g(x, t)**2 * p(x)
    return -grad(f_times_p)(x, t, p) + grad(grad(g_squared_times_p))(x, t, p)


def plot_gradient_field(ax, func, xlimits, ylimits, numticks=30):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    zs = vmap(func)(Y.ravel(), X.ravel())
    Z = zs.reshape(X.shape)
    ax.quiver(X, Y, np.ones(Z.shape), Z)
    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)


def build_fd_func(y0, y1, pvals):
    # pvals should be evaluated at linspace(y0, y1, n)

    n = pvals.shape[0]
    dx = (y1 - y0) / (n + 1)

    @custom_jvp
    def interp(x):
        xu = interval_to_unit(x, y0, y1)
        xun = np.clip(xu * n, 0, n)
        return map_coordinates(pvals, np.atleast_1d(xun), order=1, mode='nearest')

    def finite_diff_jvp(x_tan, _, x):
        return x_tan * (interp(x + dx) - interp(x - dx)) / (2. * dx)
    interp.defjvps(finite_diff_jvp)

    return interp


def unit_to_interval(y, x0, x1):
    return y * (x1 - x0) + x0


def interval_to_unit(x, y0, y1):
    return (x - y0) / (y1 - y0)


def check_discretize_grad():

    def f(y, t): return -y + np.sin(2. * y) - np.cos(2. * t)
    def g(y, t): return np.exp(np.sin(2. * y) - np.cos(2. * t))

    init_ps = lambda x: norm.pdf(x, 0., 0.8)
    exact_grad_init_ps = vmap(grad(init_ps))
    exact_grad_grad_init_ps = vmap(grad(grad(init_ps)))
    ylims = [-2.1, 2.1]
    xs = np.linspace(ylims[0], ylims[1], 10000)
    pvals = init_ps(xs)
    density1 = build_fd_func(ylims[0], ylims[1], pvals)

    density = vmap(density1)
    density_grad = vmap(grad(density1))
    density_grad_grad = vmap(grad(grad(density1)))

    ylims2 = [-2.2, 2.2]
    xs2 = np.linspace(ylims2[0], ylims2[1], 30000)

    # Set up figure.
    fig = plt.figure(figsize=(8, 6), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.ion()
    plt.show(block=False)
    plt.plot(xs2, density(xs2), 'g')
    plt.plot(xs2, density_grad(xs2), 'b')
    plt.plot(xs2, exact_grad_init_ps(xs2), 'b--')
    plt.plot(xs2, density_grad_grad(xs2), 'r')
    plt.plot(xs2, exact_grad_grad_init_ps(xs2), 'r--')

    dp_dt = lambda x, t, p: fokker_planck(f, g, x, t, p)
    print(dp_dt(np.array(0.1), 0.2, density1))
    fp = vmap(lambda x: dp_dt(x, 0., density1))
    plt.plot(xs2, fp(xs2), 'k')

    ax.set_xlabel('x')
    ax.set_ylabel('p')
    plt.draw()
    plt.pause(100)


def plot_fokker():

    def f(y, t): return -y + np.sin(2. * y) - np.cos(2. * t)
    def g(y, t): return 0.1#np.exp(np.sin(2. * y)) #- np.cos(2. * t))

    t0 = 0.1
    t1 = 0.2
    ts = np.linspace(t0, t1, 100)

    init_ps = lambda x: norm.pdf(x, 0., 0.3)
    ylims = [-2.1, 2.1]
    xs = np.linspace(ylims[0], ylims[1], 200)
    pvals = init_ps(xs)
    density1 = build_fd_func(ylims[0], ylims[1], pvals)

    #ylims2 = [-2., 2.]
    #xs2 = np.linspace(ylims2[0], ylims2[1], 3000)

    def dp_dt(x, t, p):
        return fokker_planck(f, g, x, t, p)

    # Set up figure.
    fig = plt.figure(figsize=(8, 6), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.ion()
    plt.show(block=False)
    ax.set_xlabel('x')
    ax.set_ylabel('p')

    p = init_ps(xs)
    if False:

        def dynamics(p, t, args):
            xs, ylims = args
            density1 = build_fd_func(ylims[0], ylims[1], p)
            fp = vmap(lambda x: dp_dt(x, t, density1))
            return fp(xs)
        full_density = odeint(dynamics, p, ts, (xs, ylims))

    if True:
        ps = [p]
        for i in enumerate(ts[1:]):

            dt = 0.01
            density1 = build_fd_func(ylims[0], ylims[1], p)
            t = 0. #ts[i]
            fp = vmap(lambda x: dp_dt(x, t, density1))

            # Euler steps.  Todo: replace with odeint
            p = p + dt * fp(xs)
            ps.append(p)

            #plt.plot(xs, p, 'g')
            #plt.plot(xs, fp(xs), 'b')
            #plt.draw()
            #plt.pause(100)

        full_density = np.array(ps)


    # Set up figure.
    fig = plt.figure(figsize=(8, 6), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.ion()
    plt.show(block=False)
    #plt.cla()
    plot_gradient_field(ax, f, xlimits=[t0, t1], ylimits=ylims)

    X, T = np.meshgrid(xs, ts)
    ax.contour(T, X, full_density)

    ax.set_xlabel('t')
    ax.set_ylabel('y')

    #for i in range(3):
    #    rng = random.PRNGKey(i)
    #    ys = sdeint_ito(f, g, y0, ts, rng, fargs, dt=1e-3)
    #    ax.plot(ts, ys, 'g-')

    plt.draw()
    plt.pause(100)


if __name__ == "__main__":
    check_discretize_grad()
    plot_fokker()


