"""Code for checking gradient variance of ELBO estimator
under Sticking the Landing, full Monte Carlo, and Li et al. 2020
"""

import argparse
import copy
import json
import logging
import math
import os

import numpy as np
import numpy.random as npr
import torch
import tqdm
from torch import nn, optim


def make_segmented_cosine_data():
    ts_ = np.concatenate((np.linspace(0.4, 0.6, 5), np.linspace(0.8, 1.0, 5)), axis=0)
    ys_ = np.cos(ts_ * (2. * math.pi))[:, None]
    ts = torch.tensor(ts_).float().to(device)
    ys = torch.tensor(ys_).float().to(device)
    return ts, ys

class SDE(nn.Module):
    def __init__(self, mu=0.0, sigma=1.0):
        super(SDE, self).__init__()
        self.mu = mu
        self.sigma = sigma
        net = nn.Sequential(nn.Linear(3, 400), nn.Tanh(), nn.Linear(400, 1))
        self.net = net
    def f(self, t, y):
        t = float(t) * torch.ones_like(y)
        inp = torch.cat((torch.sin(t), torch.cos(t), y), dim=-1)
        return self.net(inp)
    def g(self, t, y):  # Shared diffusion.
        return self.sigma * torch.ones_like(y)
    def h(self, t, y):
        return self.mu * y
    def f_detach(self, t, y):
        t = float(t) * torch.ones_like(y)
        inp = torch.cat((torch.sin(t), torch.cos(t), y), dim=-1)
        net = copy.deepcopy(self.net)
        return net(inp)

def em(sde, y0, ts, dt=1e-2, mode="stl"):
    y = y0
    t = ts[0]
    ys = [y0]
    logqp = torch.tensor(0.0).to(device)
    for t_next in ts[1:]:
        while True:
            if t + dt >= t_next:
                _dt = t_next - t
                exit_loop = True
            else:
                _dt = dt
                exit_loop = False
            assert _dt > 0, 'Underflow...'
            _dbt = torch.randn_like(y) * math.sqrt(_dt)
            f_eval = sde.f(t, y)
            g_eval = sde.g(t, y)
            h_eval = sde.h(t, y)
            f_eval_detached = sde.f_detach(t, y)
            y = y + f_eval * _dt + g_eval * _dbt
            t = t + _dt
            if mode == "fmc":
                term1 = .5 * ((f_eval - h_eval) / g_eval) ** 2. * _dt
                term2 = ((f_eval - h_eval) / g_eval) * _dbt
                logqp = logqp + term1.sum(1).mean(0) + term2.sum(1).mean(0)
            elif mode == "stl":
                term1 = .5 * ((f_eval - h_eval) / g_eval) ** 2. * _dt
                term2 = ((f_eval_detached - h_eval) / g_eval) * _dbt
                logqp = logqp + term1.sum(1).mean(0) + term2.sum(1).mean(0)
            elif mode == "ckl":
                term1 = .5 * ((f_eval - h_eval) / g_eval) ** 2. * _dt
                logqp = logqp + term1.sum(1).mean(0)
            else:
                raise ValueError()
            if exit_loop:
                ys.append(y)
                break
    return torch.stack(ys, dim=0), logqp

def loglikelihood(x, mean, std=0.1):
    assert x.size() == mean.size()
    px = torch.distributions.Normal(loc=mean, scale=std)
    return px.log_prob(x)

def train(sde, ts, ys, optimizer, lr_scheduler, global_step=0, mode="fmc", start_epoch=0):
    """Train with vanilla ELBO."""
    fmc_means, stl_means, ckl_means = [], [], []
    fmc_stds, stl_stds, ckl_stds = [], [], []
    y0 = ys[0:1, :].repeat(args.batch_size, 1)
    for epoch in range(start_epoch, args.stops[-1]):
        optimizer.zero_grad()
        mean, logqp = em(sde=sde, y0=y0, ts=ts, dt=args.dt, mode=mode)
        logpy = loglikelihood(ys.unsqueeze(1).repeat(1, args.batch_size, 1), mean).sum(2).mean(1).sum(0)
        elbo = logpy - logqp
        loss = -elbo
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        global_step += 1
        logging.warning(f'global_step: {global_step}, logpy: {logpy:.4f}, logqp: {logqp:.4f}')
        del elbo, loss
        if global_step in args.stops:
            fmc_grads = compute_per_sample_grads(sde=sde, ts=ts, ys=ys, mode="fmc")
            fmc_std_avg, fmc_std_norm = fmc_grads.std(0).mean().item(), fmc_grads.std(0).norm().item()  # Scalars.
            fmc_mean = fmc_grads.mean(0).detach().cpu().numpy()  # (num_params,).
            stl_grads = compute_per_sample_grads(sde=sde, ts=ts, ys=ys, mode="stl")
            stl_std_avg, stl_std_norm = stl_grads.std(0).mean().item(), stl_grads.std(0).norm().item()  # Scalars.
            stl_mean = stl_grads.mean(0).detach().cpu().numpy()  # (num_params,).
            ckl_grads = compute_per_sample_grads(sde=sde, ts=ts, ys=ys, mode="ckl")
            ckl_std_avg, ckl_std_norm = ckl_grads.std(0).mean().item(), ckl_grads.std(0).norm().item()  # Scalars.
            ckl_mean = ckl_grads.mean(0).detach().cpu().numpy()  # (num_params,).
            fmc_stds.append((fmc_std_avg, fmc_std_norm))
            stl_stds.append((stl_std_avg, stl_std_norm))
            ckl_stds.append((ckl_std_avg, ckl_std_norm))
            fmc_means.append(fmc_mean)
            stl_means.append(stl_mean)
            ckl_means.append(ckl_mean)
    fmc_means = np.array(fmc_means)
    stl_means = np.array(stl_means)
    ckl_means = np.array(ckl_means)
    fmc_stds = np.array(fmc_stds)
    stl_stds = np.array(stl_stds)
    ckl_stds = np.array(ckl_stds)

    return fmc_stds, stl_stds, ckl_stds, fmc_means, stl_means, ckl_means

def flatten_grad(module):
    return torch.cat([p.grad.flatten() for p in module.parameters() if p.grad is not None])

def count_params(module):
    return sum([p.numel() for p in module.parameters()])

def compute_per_sample_grads(sde, ts, ys, mode="fmc"):
    grads = []
    y0 = ys[0:1, :]
    for _ in tqdm.tqdm(range(1, args.N + 1)):
        sde.zero_grad()
        mean, logqp = em(sde=sde, y0=y0, ts=ts, dt=args.dt, mode=mode)
        logpy = loglikelihood(ys.unsqueeze(1), mean).sum(2).mean(1).sum(0)
        elbo = logpy - logqp
        elbo.backward()
        grads.append(flatten_grad(sde))
        del elbo
    return torch.stack(grads, dim=0)

def main():
    ts, ys = make_segmented_cosine_data()
    sde = SDE().to(device)
    optimizer = optim.Adam(lr=args.lr, params=sde.parameters())
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=.999)
    start_epoch = 0
    if os.path.exists(os.path.join(args.train_dir, f"{args.seed}_{args.ckpt_pth}")):
        ckpts = torch.load(os.path.join(args.train_dir, f"{args.seed}_{args.ckpt_pth}"))
        optimizer.load_state_dict(ckpts['optimizer'])
        sde.load_state_dict(ckpts['model'])
        start_epoch = ckpts['epoch']
    fmc_stds, stl_stds, ckl_stds, fmc_means, stl_means, ckl_means = train(
        sde=sde, ts=ts, ys=ys, optimizer=optimizer, lr_scheduler=lr_scheduler, start_epoch=start_epoch
    )
    final_path = os.path.join(args.train_dir, f'final_{args.seed}.json')
    info_dict = {
                'fmc_stds': fmc_stds.tolist(),
                'stl_stds': stl_stds.tolist(),
                'ckl_stds': ckl_stds.tolist(),
                'fmc_means': fmc_means.tolist(),
                'stl_means': stl_means.tolist(),
                'ckl_means': ckl_means.tolist(),
                'stops': args.stops
                }

    with open(final_path, 'w') as f:
        json.dump(info_dict, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-gpu', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train-dir', type=str, required=True)
    parser.add_argument('--dt', type=float, default=3e-3)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--N', type=int, default=2000)
    parser.add_argument('--stops', type=int, nargs='+', default=[1, 50, 100, 500, 1000, 2000, 3000])
    parser.add_argument('--ckpt-pth', type=str, default='state.ckpt')
    parser.add_argument('--save-ckpt', action='store_true')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_gpu else 'cpu')
    npr.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.debug:
        logging.getLogger().setLevel(logging.INFO)
    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)
    config_path = os.path.join(args.train_dir, f'config_{args.seed}.json')
    with open(config_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    main()
