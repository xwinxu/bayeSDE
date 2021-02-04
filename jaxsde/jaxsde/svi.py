import jax.numpy as np

from .sdeint import sdeint_ito


def unpack(aug_state):
    return aug_state[:-1], aug_state[1]


def pack(z, logpq):
    return np.hstack([z, logpq])


def make_aug_dynamics(prior_drift, diffusion, posterior_drift):
    # Ito formulation

    def aug_drift(aug_state, t, args):
        # Only for diagonal diffusion
        z, _ = unpack(aug_state)
        prior_params, post_params = args
        prior_drift_eval = prior_drift(z, t, prior_params)
        diffusion_eval = diffusion(z, t, prior_params)
        posterior_drift_eval = posterior_drift(z, t, post_params)
        logpq = -0.5 * ((posterior_drift_eval - prior_drift_eval)**2 / diffusion_eval) ** 2
        return pack(posterior_drift_eval, logpq)

    def aug_diffusion(aug_state, t, args):
        prior_params, post_params = args
        z, _ = unpack(aug_state)
        # Only for diagonal diffusion
        diffusion_eval = diffusion(z, t, prior_params)
        return pack(diffusion_eval, 0.)

    return aug_drift, aug_diffusion

def mc_elbo(z0, t0, t1, prior_params, post_params, log_likelihood_params,
            prior_drift, diffusion, posterior_drift, log_likelihood, rng):
    # Ito formulation
    aug_drift, aug_diffusion = make_aug_dynamics(prior_drift, diffusion, posterior_drift)
    aug_init = pack(z0, 0.)
    out = sdeint_ito(aug_drift, aug_diffusion, aug_init, np.array([t0, t1]), rng, (prior_params, post_params), dt=0.1)
    final_state, logpq = unpack(out[1])
    return logpq + log_likelihood(final_state, log_likelihood_params)
