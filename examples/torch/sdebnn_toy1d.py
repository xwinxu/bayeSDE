"""Train + plot code for non-gaussian multi-modal posteriors w/ Cauchy likelihood and OU prior.
"""
import contextlib
import math
import os
import sys

import matplotlib._color_data as mcd
import matplotlib.pyplot as plt
import numpy as np
import torchsde.diffeq_layers as diffeq_layers

import torch
import torch.nn as nn


def em_solver(f, g, f_p, z0, t):
    z_t = [z0]
    kldiv = torch.zeros(1).to(z0)
    for t0, t1 in zip(t[:-1], t[1:]):
        dt = t1 - t0
        dW = torch.randn_like(z0) * torch.sqrt(dt)

        f_t = f(t0, z_t[-1])
        g_t = g(t0, z_t[-1])
        z_t.append(z_t[-1] + f_t * dt + g_t * dW)

        u = (f_t - f_p(t0, z_t[-1])) / g_t
        kldiv = kldiv + dt * torch.abs(u * u) * 0.5
    return z_t, kldiv


def construct_odenet(dims):
    layers = []
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        layers.append(diffeq_layers.ConcatLinear(in_dim, out_dim))
        layers.append(diffeq_layers.TimeDependentSwish(out_dim))
    layers = layers[:-1]  # remove last activation
    return diffeq_layers.SequentialDiffEq(*layers)


class VariationalSDE(nn.Module):

    def __init__(self, f_func, ou_drift_coef=0.1, diff_coef=1.0 * math.sqrt(2.0)):
        super().__init__()
        self.f_func = f_func
        self.g_func = lambda t, x: diff_coef * torch.ones_like(x)

        # OU prior
        self.f_prior_func = lambda t, x: -ou_drift_coef * x

        self.start_time = 0.0
        self.end_time = 3.0

    def forward(self, nsamples):
        z0 = torch.randn(nsamples, 1) * 3
        z_t, kldiv = em_solver(self.f_func, self.g_func, self.f_prior_func, z0, torch.linspace(self.start_time, self.end_time, 100).to(z0))
        return torch.stack(z_t, dim=0), kldiv


def loglik_fn(x, z, scale=1.0):
    scale = torch.as_tensor(scale)
    dist = torch.distributions.cauchy.Cauchy(loc=z, scale=scale)
    return dist.log_prob(x)


def score_fn(x, z, scale):
    with torch.enable_grad():
        x = x.requires_grad_(True)
        loglik = loglik_fn(x, z, scale)
        score = torch.autograd.grad(loglik.sum(), x, create_graph=True)[0]
        return score

@contextlib.contextmanager
def random_seed_torch(seed):
    cpu_rng_state = torch.get_rng_state()

    np.random.seed(seed)
    torch.manual_seed(seed)

    try:
        yield
    finally:
        torch.set_rng_state(cpu_rng_state)


if __name__ == "__main__":
    output_dir = sys.argv[1]
    os.makedirs(f"/checkpoint/winniexu/{output_dir}", exist_ok=True)

    torch.manual_seed(0)

    f_func = construct_odenet([1, 64, 64, 1])

    # Increase lipschitz at initialization.
    for m in f_func.modules():
        if hasattr(m, "weight"):
            m.weight.data *= 1.0

    q_sde = VariationalSDE(f_func)

    optimizer = torch.optim.Adam(q_sde.parameters(), lr=2e-3, betas=(0.9, 0.999))

    data = [torch.tensor(5.0), torch.tensor(-5.0)]

    info = {}

    for itr in range(300):

        # Sample and compute ELBO.
        n = 20
        z_t, kldiv = q_sde(n)
        z_T = z_t[-1]

        loglik = 0.0
        for x in data:
            loglik = loglik + loglik_fn(x.to(z_T), z_T, scale=1.0)

        elbo = loglik - 0.2 / (q_sde.end_time - q_sde.start_time) * kldiv

        # Comment this line out to use ELBO instead.
        elbo = torch.logsumexp(elbo, dim=0) - math.log(n)

        elbo = elbo.mean()

        optimizer.zero_grad()
        elbo.mul(-1.0).backward()
        optimizer.step()

        # if itr % 5 == 0 or itr + 1 == 1000:
        if itr == 0 or itr == 55:
            print(itr, elbo.item())
