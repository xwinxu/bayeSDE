from jax import jvp
from jax.flatten_util import ravel_pytree
import jax.numpy as np

from .sdeint import ito_integrate
from .sde_vjp import make_milstein_prod


def make_forward_dynamics(f, g, b, unpack, tan_t0, tan_args):
    # Equation (1.3) from
    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.78.3399&rep=rep1&type=pdf

    def aug_f(augmented_state, t, args):
        y, a, args = unpack(augmented_state)
        dy_dt, da_dt = jvp(f, (y, t, args), (a, tan_t0, tan_args))
        return np.concatenate([dy_dt, da_dt])

    def aug_g(augmented_state, t, args):
        y, a = unpack(augmented_state)
        dy_dt, da_dt = jvp(g, (y, t, args), (a, tan_t0, tan_args))
        return np.concatenate([dy_dt, da_dt])

    aug_ggp_prod = make_milstein_prod(g, unpack)

    def aug_b(t):
        bval = b(t)
        return np.hstack((bval, bval))

    return aug_f, aug_g, aug_b, aug_ggp_prod


def jvp_ito_integrate(tan_y0, tan_args, tan_t0, tan_t1, y0, args, f, g, ts, b, dt):
    # Forward mode differentiation through an SDE solution.

    aug_y0, unpack = ravel_pytree((y0, tan_y0, tan_args))  # Make un-concatenate function.

    aug_f, aug_g, aug_b, aug_ggp_prod = make_forward_dynamics(f, g, b, unpack, tan_t0, tan_args)

    # Run augmented system backwards.
    aug_ans = ito_integrate(aug_f, aug_g, aug_y0, ts, aug_b, dt, args, gdg=aug_ggp_prod)
    yt, tan_yt, tan_args = unpack(aug_ans[-1])

    # TODO: Deal with tan_t1 and flattening args.
    return (np.array([y0, yt]), np.array([tan_y0, tan_yt]))
