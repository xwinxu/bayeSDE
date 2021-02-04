import functools

from jax import vjp, jacobian, jvp, grad
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map
import jax.numpy as np

from .sde_utils import time_reflect_ito, diag_jac, time_reflect_stratonovich
from .sdeint import ito_integrate, stratonovich_integrate, _sdeint_ito

from .brownian import make_brownian_motion

def make_ito_adjoint_dynamics(flat_f, flat_g, unpack):
    # Equations (1.4) and (1.6) from
    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.78.3399

    def aug_f(augmented_state, t, flat_args):
        y, y_adj, args_adj = unpack(augmented_state)

        # Evaluate drift function and its derivatives.
        fval, vjp_all = vjp(flat_f, y, t, flat_args)
        f_vjp_a, f_vjp_t, f_vjp_args = vjp_all(-y_adj)

        # Gradients due to diffusion term.
        _, vjp_g = vjp(flat_g, y, t, flat_args)
        adj_times_dgdx,   _,       _ = vjp_g(y_adj)
        g_vjp_a, g_vjp_t, g_vjp_args = vjp_g(adj_times_dgdx)

        return np.concatenate([fval,
                               f_vjp_a    + g_vjp_a,
                               f_vjp_args + g_vjp_args])

    def aug_flat_g_prod(augmented_state, t, args, v):
        # Implicitly computes r_i = sum_j sigma_ij v_j
        # where sigma represents the D x Q diffusion matrix of the
        # augmented dynamics, where D = len(y) + len(y) + len(args)
        # and Q is the dimension of the noise.
        # v is Q-dimensional
        # r is N-dimensional
        y, y_adj, arg_adj = unpack(augmented_state)
        gval, vjp_g = vjp(flat_g, y, t, args)
        vjp_a, vjp_t, vjp_args = vjp_g(-y_adj * v)
        return np.concatenate([gval * v, vjp_a, vjp_args])

    adj_flat_gdg = make_milstein_prod(flat_g, unpack)

    return aug_f, aug_flat_g_prod, adj_flat_gdg


def make_strat_adjoint_dynamics(flat_f, flat_g, unpack):

    def aug_f(augmented_state, t, flat_args):
        y, y_adj, args_adj = unpack(augmented_state)

        # Evaluate drift function and its derivatives.
        fval, vjp_all = vjp(flat_f, y, t, flat_args)
        f_vjp_a, f_vjp_t, f_vjp_args = vjp_all(-y_adj)
        return np.concatenate([fval, f_vjp_a, f_vjp_args])

    def aug_flat_g_prod(augmented_state, t, args, v):
        # Implicitly computes r_i = sum_j sigma_ij v_j
        # where sigma represents the D x Q diffusion matrix of the
        # augmented dynamics, where D = len(y) + len(y) + len(args)
        # and Q is the dimension of the noise.
        # v is Q-dimensional
        # r is N-dimensional
        y, y_adj, arg_adj = unpack(augmented_state)
        gval, vjp_g = vjp(flat_g, y, t, args)
        vjp_a, vjp_t, vjp_args = vjp_g(-y_adj * v)
        return np.concatenate([gval * v, vjp_a, vjp_args])

    adj_flat_gdg = make_milstein_prod(flat_g, unpack)

    return aug_f, aug_flat_g_prod, adj_flat_gdg


def make_explicit_sigma(g, unpack):
    # Builds the augmented diffusion function explicitly.
    # Should only be used for testing since it's inefficient.
    def explicit_sigma(aug_y, t, args):
        y, y_adj, arg_adj = unpack(aug_y)
        gval = g(y, t, args)
        jac_g, jac_args = jacobian(g, argnums=(0, 2))(y, t, args)
        sigma_adj = jac_g * -y_adj
        sigma_theta = jac_args.T * -y_adj
        return np.vstack([np.diag(gval), sigma_adj, sigma_theta])
    return explicit_sigma


def make_milstein_prod(g, unpack):

    def adjoint_milstein_prod(aug_state, t, args, v):
        # Computes Milstein correction factor:
        # a vector m of size |c|
        # where m_i = sum_j,l (partial sigma_ij / partial c_l) * sigma_l,j * v_j,
        # where sigma is the |c| * |noise| dimensional function that computes the adjoint dynamics
        # as a function of the augmented state c = [y, adj_, adj_theta]
        y, adj, adj_args = unpack(aug_state)

        gval, vjp_all = vjp(g, y, t, args)
        gdg_times_v, _, _ = vjp_all(gval * v)

        dgdx = diag_jac(g, y, t, args)

        # Compute (a * v * dgdx)^T dg / d(x, args)
        prod_partials_adj, _, prod_partials_args = vjp_all(adj * v * dgdx)

        def gdg(y, t, args, v):
            g_y_only = lambda y: g(y, t, args)
            gval, tangent = jvp(g_y_only, (y,), (v,))
            return np.sum(tangent)

        gdg_v = lambda y, t, args: gdg(y, t, args, adj * v * gval)

        # Compute d/d_(x,args) (sum_j a_j * v_j * g_j * dg_j/dx_j
        mixed_partials_adj, mixed_partials_args = grad(gdg_v, argnums=(0, 2))(y, t, args)

        return np.concatenate([gdg_times_v, prod_partials_adj - mixed_partials_adj,
                                            prod_partials_args - mixed_partials_args])

    return adjoint_milstein_prod


def make_explicit_milstein(sigma, y, t, args):
    # Builds the milstein correction term explicitly.
    # Should only be used for testing since it's inefficient.
    # For a matrix-valued function sigma: D -> D x Q, computes
    # m_ij = sum_l (d sigma_ij / d y_l) sigma_lj
    # where
    # y is a D x 1 vector
    # m is a D x Q matrix
    sigmaval = sigma(y, t, args).T[np.newaxis, :, :]  # 1 x Q x D
    jac = jacobian(sigma)(y, t, args)                 # D x Q x D (out x in)
    return np.sum(jac * sigmaval, axis=2)             # D x Q  (sum over in)


def vjp_integrate(v_yt, v_argst, yt, f, g, ts, bm, dt, args,
                  make_adjoint_dynamics, integrate, time_reflect):
    """Compute the gradient by simulating the vjp dynamics backward in time.

    Args:
        v_yt: gradient of loss w.r.t. output at terminal time
        v_argst: gradient of loss w.r.t. parameters at terminal time; for applications other than time-series modeling,
            this should set to be `np.zeros_like(flat_args)`
        yt: state at terminal time
    The meaning of other arguments should be clear from context.

    Returns:
        A `np.ndarray` for reconstructed state at the initial time and a tuple of state adjoint and parameter adjoint.
    """
    # Flatten args into a vector.
    flat_args, unravel = ravel_pytree(args)
    def flat_f(y, t, flat_args): return f(y, t, unravel(flat_args))
    def flat_g(y, t, flat_args): return g(y, t, unravel(flat_args))

    # Construct adjoint system.
    aug_yt1, unpack = ravel_pytree((yt, v_yt, v_argst))
    rf, rg, rb, rts = time_reflect(flat_f, flat_g, bm, ts)
    aug_f, aug_g_prod, aug_ggp = make_adjoint_dynamics(rf, rg, unpack)

    # Run augmented system backwards.
    aug_ans = integrate(aug_f, None, aug_yt1, rts, rb, dt, flat_args, gdg=aug_ggp, g_prod=aug_g_prod)
    y0_rec, y_adj, arg_adj = unpack(aug_ans[-1])
    return y0_rec, (y_adj, arg_adj)


def vjp_ito_integrate(v_yt, v_argst, yt, f, g, ts, bm, dt, args=(), method='milstein'):
    integrate_fn = functools.partial(ito_integrate, method=method)
    return vjp_integrate(
        v_yt, v_argst, yt, f, g, ts, bm, dt, args,
        make_ito_adjoint_dynamics, integrate_fn, time_reflect_ito)


def vjp_strat_integrate(v_yt, v_argst, yt, f, g, ts, bm, dt, args=(), method='milstein'):
    integrate_fn = functools.partial(stratonovich_integrate, method=method)
    return vjp_integrate(
        v_yt, v_argst, yt, f, g, ts, bm, dt, args,
        make_strat_adjoint_dynamics, integrate_fn, time_reflect_stratonovich)


def _sdeint_ito_rev(f, g, dt, method, rep, res, grads):
    ys, ts, rng, args = res

    # Could avoid recreating BM here if we could figure out tracing problem.
    b = make_brownian_motion(ts[0], np.zeros(ys[-1].shape), ts[-1], rng, rep=rep)

    v_yt = grads[-1]
    v_argst = tree_map(np.zeros_like, args)

    y0_rec, exact_grad = vjp_ito_integrate(v_yt, v_argst, ys[-1], f, g, ts, b, dt, args, method=method)
    (y_adj, arg_adj) = exact_grad

    return np.zeros_like(rng), y_adj, np.zeros_like(ts), arg_adj


def _sdeint_ito_fwd(f, g, dt, method, rng, rep, y0, ts, args):
    ys = _sdeint_ito(f, g, dt, method, rng, rep, y0, ts, args)
    return ys, (ys, ts, rng, args)
_sdeint_ito.defvjp(_sdeint_ito_fwd, _sdeint_ito_rev)
