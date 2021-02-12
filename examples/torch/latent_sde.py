"""Training a latent SDE model to lern bi-modal outputs.
"""
import argparse
import json
import logging
import math
import os
import random
from collections import namedtuple
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torchsde
import tqdm

import torch
from torch import distributions, nn, optim
from torch.utils.tensorboard import SummaryWriter

# w/ underscore -> numpy; w/o underscore -> torch.
Data = namedtuple('Data', ['ts_', 'ts_ext_', 'ts_vis_', 'ts', 'ts_ext', 'ts_vis', 'ys', 'ys_'])
POOL_SIZE = 16
CACHE_SIZE = 500


def write_config(args: argparse.Namespace):
    os.makedirs(args.train_dir, exist_ok=True)
    os.makedirs(os.path.join(args.train_dir, 'ckpts'), exist_ok=True)
    os.makedirs(os.path.join(args.train_dir, 'plots'), exist_ok=True)
    config_path = os.path.join(args.train_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(args.__dict__, f, indent=4)
    logging.warning(f"Wrote config: {config_path}")


class ConstantScheduler(object):
    def __init__(self, constant=1.0):
        self._constant = constant
        self._val = constant

    def step(self, current_itr):
        self._val = self._constant

    def set(self, x):
        self._val = x

    @property
    def val(self):
        return self._val


class LinearScheduler(object):
    def __init__(self, iters, maxval=1.0):
        self._iters = max(1, iters)
        self._val = maxval / self._iters
        self._maxval = maxval

    def step(self):
        self._val = min(self._maxval, self._val + self._maxval / self._iters)

    def set(self, x):
        self._val = x

    @property
    def val(self):
        return self._val


class LinearDecayScheduler(object):
    def __init__(self, iters, maxval=100.0):
        self._iters = max(1, iters)
        self._val = maxval
        self._minval = maxval / self._iters

    def step(self):
        self._val = max(self._minval, self._val - self._minval)

    def set(self, x):
        self._val = x

    @property
    def val(self):
        return self._val


class HalfwayScheduler(object):
    def __init__(self, iters=5000, maxval=1.0, halfway_val=100.):
        self._iters = max(1, iters)
        self._val = maxval
        self._maxval = maxval
        self._halfway_val = halfway_val

    def step(self, current_itr):
        halfway = self._iters // 2
        self._val = self._maxval if current_itr < halfway else self._halfway_val

    def set(self, x):
        self._val = x

    @property
    def val(self):
        return self._val


class EMAMetric(object):
    def __init__(self, gamma: Optional[float] = .99):
        super(EMAMetric, self).__init__()
        self._val = 0.
        self._gamma = gamma

    def step(self, x: Union[torch.Tensor, np.ndarray]):
        x = x.detach().cpu().numpy() if torch.is_tensor(x) else x
        self._val = self._gamma * self._val + (1 - self._gamma) * x
        return self._val

    def set(self, x: Union[torch.Tensor, np.ndarray]):
        self._val = x

    @property
    def val(self):
        return self._val


def str2bool(v):
    """Used for boolean arguments in argparse; avoiding `store_true` and `store_false`."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def make_optimizer(optimizer, params, lr):
    optimizer_constructor = {
        "adam": optim.Adam,
        "adamax": optim.Adamax,
        "adadelta": optim.Adadelta,
        "adagrad": optim.Adagrad,
        "sgd": optim.SGD
    }[optimizer]
    return optimizer_constructor(params=params, lr=lr)


def manual_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _stable_division(a, b, epsilon=1e-7):
    b = torch.where(b.abs().detach() > epsilon, b, torch.full_like(b, fill_value=epsilon) * b.sign())
    return a / b


class LatentSDE(torchsde.SDEIto):

    def __init__(self, theta=1.0, mu=0.0, sigma=0.5):
        super(LatentSDE, self).__init__(noise_type="diagonal")
        logvar = math.log(sigma ** 2 / (2. * theta))

        # Prior drift.
        self.register_buffer("theta", torch.tensor([[theta]]))
        self.register_buffer("mu", torch.tensor([[mu]]))
        self.register_buffer("sigma", torch.tensor([[sigma]]))

        # p(y0).
        self.register_buffer("py0_mean", torch.tensor([[mu]]))
        self.register_buffer("py0_logvar", torch.tensor([[logvar]]))

        # Approximate posterior drift: Takes in 2 positional encodings and the state.
        self.net = nn.Sequential(
            nn.Linear(2, 200),
            nn.Softplus(),
            nn.Linear(200, 200),
            nn.Softplus(),
            nn.Linear(200, 1)
        )
        # Initialization trick from Glow.
        self.net[-1].weight.data.fill_(0.)
        self.net[-1].bias.data.fill_(0.)

        # q(y0).
        self.qy0_mean = nn.Parameter(torch.tensor([[mu]]), requires_grad=False)
        self.qy0_logvar = nn.Parameter(torch.tensor([[logvar]]), requires_grad=False)

    def f(self, t, y):  # Approximate posterior drift.
        return self.net(torch.cat((t.expand_as(y[:, :1]), y), dim=-1))

    def g(self, t, y):  # Shared diffusion.
        return self.sigma.repeat(y.size(0), 1)

    def h(self, t, y):  # Prior drift.
        return self.theta * (self.mu - y)

    def f_aug(self, t, y):  # Drift for augmented dynamics with logqp term.
        y = y[:, 0:1]
        f, g, h = self.f(t, y), self.g(t, y), self.h(t, y)
        u = _stable_division(f - h, g)
        f_logqp = .5 * (u ** 2).sum(dim=1, keepdim=True)
        return torch.cat([f, f_logqp], dim=1)

    def g_aug(self, t, y):  # Diffusion for augmented dynamics with logqp term.
        y = y[:, 0:1]
        g = self.g(t, y)
        g_logqp = torch.zeros_like(y)
        return torch.cat([g, g_logqp], dim=1)

    def forward(self, ts, batch_size, eps=None):
        eps = torch.randn(batch_size, 1).to(self.qy0_std) if eps is None else eps
        y0 = self.qy0_mean + eps * self.qy0_std
        qy0 = distributions.Normal(loc=self.qy0_mean, scale=self.qy0_std)
        py0 = distributions.Normal(loc=self.py0_mean, scale=self.py0_std)
        logqp0 = distributions.kl_divergence(qy0, py0).sum(dim=1)  # KL(t=0).

        bm = torchsde.BrownianInterval(
            t0=ts[0],
            t1=ts[-1],
            dtype=y0.dtype,
            device=y0.device,
            size=(batch_size, 2),
            pool_size=POOL_SIZE,
            cache_size=CACHE_SIZE
        )
        aug_y0 = torch.cat([y0, torch.zeros(batch_size, 1).to(y0)], dim=1)
        aug_ys = sdeint_fn(
            sde=self,
            bm=bm,
            y0=aug_y0,
            ts=ts.to(device),
            method=args.method,
            dt=args.dt,
            adaptive=args.adaptive,
            adjoint_adaptive=args.adjoint_adaptive,
            rtol=args.rtol,
            atol=args.atol,
            names={'drift': 'f_aug', 'diffusion': 'g_aug'}
        )
        ys, logqp_path = aug_ys[:, :, 0:1], aug_ys[-1, :, 1]
        logqp = (logqp0 + logqp_path).mean(dim=0)  # KL(t=0) + KL(path).
        return ys, logqp

    def sample_p(self, ts, batch_size, eps=None, bm=None):
        eps = torch.randn(batch_size, 1).to(self.py0_mean) if eps is None else eps
        y0 = self.py0_mean + eps * self.py0_std
        # TODO:
        return sdeint_fn(self, y0, ts.to(device), bm=bm, method='euler', dt=args.dt, names={'drift': 'h'})

    def sample_q(self, ts, batch_size, eps=None, bm=None):
        eps = torch.randn(batch_size, 1).to(self.qy0_mean) if eps is None else eps
        y0 = self.qy0_mean + eps * self.qy0_std
        # TODO:
        return sdeint_fn(self, y0, ts.to(device), bm=bm, method='euler', dt=args.dt)

    @property
    def py0_std(self):
        return torch.exp(.5 * self.py0_logvar)

    @property
    def qy0_std(self):
        return torch.exp(.5 * self.qy0_logvar)


def make_segmented_cosine_data():
    ts0_, ts1_ = np.linspace(0.3, 0.6, 2)[:0], np.linspace(1.40, 1.5, 1)[:0]
    ys0_, ys1_ = tuple(np.cos(t * (2. * math.pi)) for t in (ts0_, ts1_))
    ts_ = np.concatenate((ts0_, np.array([1.0]), ts1_))
    ts_ext_ = np.array([0.4] + list(ts_) + [1.6])
    ts_vis_ = np.linspace(0.4, 1.6, 300)
    ys_ = np.stack(
        (
            np.concatenate((ys0_, np.array([-1.41]), ys1_)),
            np.concatenate((ys0_, np.array([-0.21]), ys1_)),
        ), axis=1
    )

    ts = torch.tensor(ts_).float()
    ts_ext = torch.tensor(ts_ext_).float()
    ts_vis = torch.tensor(ts_vis_).float()
    ys = torch.tensor(ys_).float().to(device)
    return Data(ts_, ts_ext_, ts_vis_, ts, ts_ext, ts_vis, ys, ys_)


def make_irregular_sine_data():
    ts_ = np.sort(np.random.uniform(low=0.4, high=1.6, size=16))
    ts_ext_ = np.array([0.] + list(ts_) + [2.0])
    ts_vis_ = np.linspace(0., 2.0, 300)
    ys_ = np.sin(ts_ * (2. * math.pi))[:, None] * 0.8

    ts = torch.tensor(ts_).float()
    ts_ext = torch.tensor(ts_ext_).float()
    ts_vis = torch.tensor(ts_vis_).float()
    ys = torch.tensor(ys_).float().to(device)
    return Data(ts_, ts_ext_, ts_vis_, ts, ts_ext, ts_vis, ys, ys_)


def make_data():
    data_constructor = {
        'segmented_cosine': make_segmented_cosine_data,
        'irregular_sine': make_irregular_sine_data
    }[args.data]
    return data_constructor()


def main():
    # Dataset.
    ts_, ts_ext_, ts_vis_, ts, ts_ext, ts_vis, ys, ys_ = make_data()
    summary = SummaryWriter(os.path.join(args.train_dir, 'tb'))

    # Plotting parameters.
    vis_batch_size = 1024
    ylims = (-1.75, 1.75)
    alphas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    percentiles = [0.999, 0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    vis_idx = np.random.permutation(vis_batch_size)
    if args.color == "blue":
        fill_color = '#9ebcda'
        mean_color = '#4d004b'
        num_samples = 60
    else:
        sample_colors = ('#fc4e2a', '#e31a1c', '#bd0026')
        fill_color = '#fd8d3c'
        mean_color = '#800026'
        num_samples = len(sample_colors)

    eps = torch.randn(vis_batch_size, 1).to(device)  # Fix seed for the random draws used in the plots.
    bm = torchsde.BrownianInterval(
        t0=ts_vis[0],
        t1=ts_vis[-1],
        size=(vis_batch_size, 1),
        device=device,
        levy_area_approximation='space-time',
        pool_size=POOL_SIZE,
        cache_size=CACHE_SIZE,
    )  # We need space-time Levy area to use the SRK solver

    # Model.
    # Note: This `mu` is selected based on the yvalue of the two endpoints of the left and right segments.
    model = LatentSDE(mu=-0.80901699, sigma=args.sigma).to(device)
    optimizer = make_optimizer(optimizer=args.optimizer, params=model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=.99997)
    kl_scheduler = LinearScheduler(iters=args.kl_anneal_iters, maxval=args.kl_coeff)
    nll_scheduler = ConstantScheduler(constant=args.nll_coef)

    logpy_metric = EMAMetric()
    kl_metric = EMAMetric()
    loss_metric = EMAMetric()

    if os.path.exists(os.path.join(args.train_dir, 'ckpts', f'state.ckpt')):
        logging.info("Loading checkpoints...")
        checkpoint = torch.load(os.path.join(args.train_dir, 'ckpts', f'state.ckpt'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        try:
            logpy_metric.set(checkpoint['logpy_metric'])
            kl_metric.set(checkpoint['kl_metric'])
            loss_metric.set(checkpoint['loss_metric'])
        except:
            logging.warning(f"Could not successfully load logpy, kl, and loss metrics from checkpoint")
        logging.info(f"Successfully loaded checkpoints at global_step {checkpoint['global_step']}")

    if args.show_prior:
        with torch.no_grad():
            zs = model.sample_p(ts=ts_vis, batch_size=vis_batch_size, eps=eps, bm=bm).squeeze()
            ts_vis_, zs_ = ts_vis.cpu().numpy(), zs.cpu().numpy()
            zs_ = np.sort(zs_, axis=1)

            img_dir = os.path.join(args.train_dir, 'prior.png')
            
            plt.subplot(frameon=False)
            for alpha, percentile in zip(alphas, percentiles):
                idx = int((1 - percentile) / 2. * vis_batch_size)
                zs_bot_ = zs_[:, idx]
                zs_top_ = zs_[:, -idx]
                plt.fill_between(ts_vis_, zs_bot_, zs_top_, alpha=alpha, color=fill_color)

            # `zorder` determines who's on top; the larger the more at the top.
            plt.scatter(ts_, ys_[:, 0], marker='x', zorder=3, color='k', s=35)  # Data.
            if args.data != "irregular_sine":
                plt.scatter(ts_, ys_[:, 1], marker='x', zorder=3, color='k', s=35)  # Data.
            plt.ylim(ylims)
            plt.xlabel('$t$')
            plt.ylabel('$Y_t$')
            plt.tight_layout()
            plt.savefig(img_dir, dpi=args.dpi)
            summary.add_figure('Prior', plt.gcf(), 0)
            logging.info(f'Prior saved to tensorboard')
            plt.close()
            logging.info(f'Saved prior figure at: {img_dir}')

    for global_step in tqdm.tqdm(range(args.train_iters)):
        # Plot and save.
        if global_step % args.pause_iters == 0 or global_step == (args.train_iters - 1):
            img_path = os.path.join(args.train_dir, "plots", f'global_step_{global_step}.png')

            with torch.no_grad():
                # TODO:
                zs = model.sample_q(ts=ts_vis, batch_size=vis_batch_size, eps=None, bm=bm).squeeze()
                samples = zs[:, vis_idx]
                ts_vis_, zs_, samples_ = ts_vis.cpu().numpy(), zs.cpu().numpy(), samples.cpu().numpy()
                zs_ = np.sort(zs_, axis=1)
                plt.subplot(frameon=False)

                if args.show_percentiles:
                    for alpha, percentile in zip(alphas, percentiles):
                        idx = int((1 - percentile) / 2. * vis_batch_size)
                        zs_bot_, zs_top_ = zs_[:, idx], zs_[:, -idx]
                        plt.fill_between(ts_vis_, zs_bot_, zs_top_, alpha=alpha, color=fill_color)

                if args.show_mean:
                    plt.plot(ts_vis_, zs_.mean(axis=1), color=mean_color)

                if args.show_samples:
                    for j in range(num_samples):
                        plt.plot(ts_vis_, samples_[:, j], linewidth=1.0)

                if args.show_arrows:
                    t_start, t_end = ts_vis_[0], ts_vis_[-1]
                    num, dt = 12, 0.12
                    t, y = torch.meshgrid(
                        [torch.linspace(t_start, t_end, num).to(device), torch.linspace(*ylims, num).to(device)]
                    )
                    t, y = t.reshape(-1, 1), y.reshape(-1, 1)
                    fty = model.f(t=t, y=y).reshape(num, num)
                    dt = torch.zeros(num, num).fill_(dt).to(device)
                    dy = fty * dt
                    dt_, dy_, t_, y_ = dt.cpu().numpy(), dy.cpu().numpy(), t.cpu().numpy(), y.cpu().numpy()
                    plt.quiver(t_, y_, dt_, dy_, alpha=0.3, edgecolors='k', width=0.0035, scale=50)

                if args.hide_ticks:
                    plt.xticks([], [])
                    plt.yticks([], [])

                plt.scatter(ts_, ys_[:, 0], marker='x', zorder=3, color='k', s=35)  # Data.
                if args.data != "irregular_sine":
                    plt.scatter(ts_, ys_[:, 1], marker='x', zorder=3, color='k', s=35)  # Data.
                plt.ylim(ylims)
                plt.xlabel('$t$')
                plt.ylabel('$Y_t$')
                plt.tight_layout()
                if global_step % args.save_fig == 0:
                    plt.savefig(img_path, dpi=args.dpi)
                current_fig = plt.gcf()
                summary.add_figure('Predictions plot', current_fig, global_step)
                logging.info(f'Predictions plot saved to tensorboard')
                plt.close()
                logging.info(f'Saved figure at: {img_path}')

                if args.save_ckpt:
                    torch.save(
                        {'model': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict()},
                        os.path.join(args.train_dir, 'ckpts', f'global_step_{global_step}.ckpt')
                    )
                    # for preemption
                    torch.save(
                        {'model': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(),
                         'global_step': global_step,
                         'logpy_metric': logpy_metric.val,
                         'kl_metric': kl_metric.val,
                         'loss_metric': loss_metric.val},
                        os.path.join(args.train_dir, 'ckpts', f'state.ckpt')
                    )

        # Train.
        optimizer.zero_grad()
        zs, kl = model(ts=ts_ext, batch_size=args.batch_size)
        zs = zs.squeeze()
        zs = zs[1:-1]  # Drop first and last which are only used to penalize out-of-data region and spread uncertainty.

        likelihood_constructor = {
            "laplace": distributions.Laplace, "normal": distributions.Normal, "cauchy": distributions.Cauchy
        }[args.likelihood]
        likelihood = likelihood_constructor(loc=zs, scale=args.scale)

        # Proper summation of log-likelihoods.
        logpy = 0.
        ys_split = ys.split(split_size=1, dim=-1)
        for _ys in ys_split:
            logpy = logpy + likelihood.log_prob(_ys).sum(dim=0).mean(dim=0)
        logpy = logpy / len(ys_split)

        loss = -logpy * nll_scheduler.val + kl * kl_scheduler.val
        loss.backward()

        optimizer.step()
        scheduler.step()
        kl_scheduler.step()
        nll_scheduler.step(global_step)

        logpy_metric.step(logpy)
        kl_metric.step(kl)
        loss_metric.step(loss)

        logging.info(
            f'global_step: {global_step}, '
            f'logpy: {logpy_metric.val:.3f}, '
            f'kl: {kl_metric.val:.3f}, '
            f'loss: {loss_metric.val:.3f}'
        )
        summary.add_scalar('KL Schedler', kl_scheduler.val, global_step)
        summary.add_scalar('NLL Schedler', nll_scheduler.val, global_step)
        summary.add_scalar('Loss', loss_metric.val, global_step)
        summary.add_scalar('KL', kl_metric.val, global_step)
        summary.add_scalar('Log(py) Likelihood', logpy_metric.val, global_step)
        logging.info(f'Logged loss, kl, logpy to tensorboard')

    summary.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-gpu', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--debug', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train-dir', type=str, required=True)
    parser.add_argument('--save-ckpt', type=str2bool, default=False, const=True, nargs="?")

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--optimizer', type=str, default="adam",
                        choices=["adam", "adadelta", "adamax", "sgd", "adagrad"])
    parser.add_argument('--data', type=str, default='segmented_cosine', choices=['segmented_cosine', 'irregular_sine'])
    parser.add_argument('--kl-anneal-iters', type=int, default=100, help='Number of iterations for linear KL schedule.')
    parser.add_argument('--kl-coeff', type=float, default=1.)
    parser.add_argument('--nll-coef', type=float, default=1.) 
    parser.add_argument('--nll-decay-coef', type=float, default=100.) 
    parser.add_argument('--train-iters', type=int, default=5000, help='Number of iterations for training.')
    parser.add_argument('--pause-iters', type=int, default=50, help='Number of iterations before pausing.')
    parser.add_argument('--save-fig', type=int, default=200, help='Number of iterations before pausing.')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size for training.')
    parser.add_argument('--likelihood', type=str, choices=['normal', 'laplace', 'cauchy'], default='cauchy')
    parser.add_argument('--scale', type=float, default=0.05, help='Scale parameter of Normal, Laplace, and Cauchy.')
    parser.add_argument('--sigma', type=float, default=0.5, help="Diffusion constant of prior process.")

    parser.add_argument('--adjoint', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--adaptive', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--adjoint-adaptive', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--method', type=str, default='euler', choices=('euler', 'milstein', 'srk'),
                        help='Name of numerical solver.')
    parser.add_argument('--dt', type=float, default=1e-2)
    parser.add_argument('--rtol', type=float, default=1e-3)
    parser.add_argument('--atol', type=float, default=1e-3)

    parser.add_argument('--show-prior', type=str2bool, default=True, const=True, nargs="?")
    parser.add_argument('--show-samples', type=str2bool, default=True, const=True, nargs="?")
    parser.add_argument('--show-percentiles', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--show-arrows', type=str2bool, default=True, const=True, nargs="?")
    parser.add_argument('--show-mean', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--hide-ticks', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--color', type=str, default='blue', choices=('blue', 'red'))
    args = parser.parse_args()

    write_config(args)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_gpu else 'cpu')
    manual_seed(args.seed)

    if args.debug:
        logging.getLogger().setLevel(logging.INFO)

    sdeint_fn = torchsde.sdeint_adjoint if args.adjoint else torchsde.sdeint

    main()

    print("finished running :)")
