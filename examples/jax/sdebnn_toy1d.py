"""
1D Regression demo with SDELayer.
"""
from __future__ import absolute_import

import argparse
import collections
import os
import sys
from functools import partial

from brax._impl.layers import SDELayer, Const, Affine, AugmentedLayer, shape_dependent, Additive, ConcatSquashLineartx, DiffEqWrappertx
from brax.utils.utils import he_normal_stdev
from brax.utils.registry import add_loss, get_activn, get_data, get_loss
import brax._impl.layers as diffeq_layers
import brax.utils.datasets as ds

import jax
import jax.numpy as jnp
import jax.random as random
from jax.api import jit, value_and_grad, vmap
from jax.experimental import optimizers, stax
from jax.flatten_util import ravel_pytree
from jax.nn.initializers import zeros

import collections
from functools import partial

def small_init(key, shape, dtype=jnp.float32):
    return zeros(key, shape, dtype=dtype) + 0.001

def mlp(rng, x_dim, hidden_dim, activn, setting):
    """
    Args:
        x_dim: output shape, i.e. D = 1 (regression) = 2 (classific'n)
        hidden_dim: drift hidden layer dimension
        activn: activation function (str) from registry
        setting: for experimental purposes, whether to turn off kl, diff, drift
    Return:
        Neural network using Stax layers
        with drift f(w,x) as state. 
        Augmented drift f(w) and diffusion g(w) of approx posteriors over weights
    """
    prior_driftw_layer = DiffEqWrappertx(Const(0.))
    if setting['prior_driftw'] == 'ou':
        prior_driftw_layer = DiffEqWrappertx(Affine(-1, 0.)) # OU process

    augmented_x_dim = x_dim + setting['aug_dim']
    activ_fn = get_activn(activn)
    if activn == 'swish':
        activ_fn = activ_fn(hidden_dim)
    def make_driftx_layer(input_shape):
        driftx_layer = stax.serial(
            ConcatSquashLineartx(hidden_dim),
            DiffEqWrappertx(activ_fn),
            ConcatSquashLineartx(hidden_dim),
            DiffEqWrappertx(activ_fn),
            ConcatSquashLineartx(input_shape[-1])
        ) 
        return driftx_layer
    driftx_layer = shape_dependent(make_driftx_layer)
    
    driftx_init_fn, _ = driftx_layer
    output_shape, w_init = driftx_init_fn(rng, (-1, augmented_x_dim)) # +1 time
    assert output_shape[0] == -1
    flat_w, _ = ravel_pytree(w_init)
    flat_w_dim = flat_w.size
    print(f"parameters in the w_hypernetwork: {flat_w.size}")

    if not setting['driftw']:
        driftw_layer = DiffEqWrappertx(Const(0.))
    else:
        def make_driftw_layer(input_shape):
            final_layer = diffeq_layers.ConcatSquashLineartx(input_shape[-1], W_init=zeros, b_init=zeros)
            if setting['stop_grad'] and setting['diff_drift']:
                final_layer = diffeq_layers.ConcatSquashLineartx(input_shape[-1]) 
            driftw_layer = stax.serial(
                ConcatSquashLineartx(hidden_dim),
                DiffEqWrappertx(activ_fn),
                ConcatSquashLineartx(hidden_dim),
                DiffEqWrappertx(activ_fn),
                final_layer,
            )
            return driftw_layer
        driftw_layer = shape_dependent(make_driftw_layer)
        if setting['diff_drift']:
            driftw_layer = Additive(prior_driftw_layer, driftw_layer)
    
    if not setting['diffw']:
        diffusion_layer = DiffEqWrappertx(Const(0.))
    elif setting['diff_const']:
        diffusion_layer = DiffEqWrappertx(Const(setting['diff_const']))
        if setting['priorw0_sigma'] >= 0:
            rng, diffkey = random.split(rng)
            def make_diffusion_layer(input_shape):
                diff_sigma = he_normal_stdev(diffkey, (input_shape[-1], 1))
                diffusion_layer = DiffEqWrappertx(Const(diff_sigma))
                return diffusion_layer
            diffusion_layer = shape_dependent(make_diffusion_layer)

    augmented_layer = stax.serial(
        AugmentedLayer(setting['aug_dim']),
        SDELayer(driftx_layer, prior_driftw_layer, driftw_layer, diffusion_layer, augmented_x_dim, flat_w_dim, setting),
    )
    return augmented_layer, flat_w_dim

@add_loss('mse')
def mse_loss(preds, targets, noise_std=3):
    mse = jnp.mean(jnp.sum((targets - preds) ** 2, axis=-1)) / (2 * noise_std ** 2) # scaling: 2 * variance noise
    assert mse.shape == ()
    return mse

@add_loss('laplace')
def laplace_loss(preds, targets, noise_std=3):
    laplace = jnp.mean(jnp.sum(jnp.abs(targets - preds), axis=-1)) # no scaling
    assert laplace.shape == ()
    return laplace

def test_loss(params, batch, apply_fn, loss_fn, rng, noise_std):
    x0, x1 = batch
    preds, _, _, _, _, _ = apply_fn(params, x0, rng, False)
    preds = preds[..., -1][..., None]
    assert preds.shape[-1] == x1.shape[-1]
    loss_ = get_loss(loss_fn)(preds, x1, noise_std)
    return loss_

def elbo(params, batch, apply_fn, rng, kl_scale, loss_fn, num_samples=10, noise_std=0.1):
    inputs, targets = batch
    rngs = jax.random.split(rng, num_samples)
    preds, ws, kl, _, _, _ = vmap(apply_fn, (None, None, 0, None))(params, inputs, rngs, False)
    targets = targets[None]
    preds = preds[..., -1][..., None]
    assert preds.shape[-1] == targets.shape[-1]
    neg_log_likelihood = get_loss(loss_fn)(preds, targets, noise_std)

    elbo_ = neg_log_likelihood + kl.mean() * kl_scale

    return elbo_

def parse_args(args):
    _DS = ['b40', 'b40gap', 'b20', 'cos', 'cos2', 'coscos', 'mnist', 'mnist2', 'cifar']
    _ACTIVS = ['relu', 'rbf', 'elu', 'softplus', 'swish', 'swish_nobeta', 'tanh'] # activations
    _LOSSES = ['mse', 'laplace', 'ce'] # likelihood functions
    _PRIORS = ['0', 'ou']
    _DIFFUSNS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1., 5e-1, 2e-1, 4e-1, 3e-1] # diffusion small constant
    _KL = [1., 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    _ALPHAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2] # sgd
    _DRIFTWS = [1e-2, 1e-1, 1.] # driftw inits

    parser = argparse.ArgumentParser(...)

    parser.add_argument("--no_batch", action='store_true', help="whether to use batching in training, default False")
    parser.add_argument("--batch_size", default=100, type=int, help="batch size 1 is default")
    parser.add_argument("--test_batch_size", default=200, type=int, help="test batch size (mnist) 1 is default")
    parser.add_argument("--data_size", default=200, type=int, help="1000 datapoints by default")
    parser.add_argument("--epochs", default=2000, type=int, help="1000 by default.")
    parser.add_argument("--aug_dim", default=0, type=int, help="Number of dimensions to augment ODE hidden state. Default 0")
    parser.add_argument("--warmup_itr", default=1e-3, type=float, help="warmup for lr schedule. 1e-3 by default.")
    parser.add_argument("--seed", default=0, type=int, help="random seed 0 by default.")
    parser.add_argument("--lr", default=1e-3, type=float, choices=_ALPHAS, help="learning rate 0.001 by default")
    parser.add_argument("--num_samples", default=100, type=int, help="number of plot samples (weights) 20 by default.")
    parser.add_argument("--hidden_dim", default=64, type=int, help="num units in hidden layer 20 by default.")
    parser.add_argument("--activn", default='swish', type=str, choices=_ACTIVS, help="activation function to use with network.")
    parser.add_argument("--loss", default='laplace', type=str, choices=_LOSSES, help="predictive likelihood loss function.")
    parser.add_argument("--ds", default='b20', type=str, choices=_DS, help="dataset to train on.")
    parser.add_argument("--num_starts", default=7, type=int, help="num starting states 7 by default.")
    parser.add_argument("--num_grad_samples", default=1000, type=int, help="num gradient samples 1000 by default.")
    parser.add_argument("--grad_bs", default=10, type=int, help="num gradient samples batched size default 10.")
    parser.add_argument("--no_kl", action="store_false", help="whether to set kl to 0")
    parser.add_argument("--no_drift", action="store_false", help="whether to set drift w to 0")
    parser.add_argument("--no_diff", action="store_false", help="whether to set diffusion w to 0")
    parser.add_argument("--stop_grad", action="store_true", help="whether to stick the landing, default no stl")
    parser.add_argument("--compare_stl", action="store_true", help="whether to plot both stl and non-stl runs on tensorboard")
    parser.add_argument("--no_diff_drift", action="store_false", help="whether to train/learn the difference between prior and posterior drifts")
    parser.add_argument("--diff_const", default=1e-1, type=float, choices=_DIFFUSNS, help="small diffusion constant")
    parser.add_argument("--prior_dw", default='ou', type=str, choices=_PRIORS, help="prior process")
    parser.add_argument("--priorw0_sigma", default=-1., type=float, help="prior on initial value w0 mu 0 and stdev as specified. Any negative if use point est for w0")
    parser.add_argument("--driftw_scale", default=1e-1, type=float, choices=_DRIFTWS, help="scale driftw init by factor and set last layer to 0")
    parser.add_argument("--kl_scale", default=1., choices=_KL, type=float, help="small scaling factor for kl in elbo loss")
    parser.add_argument("--driftx_init", default=-1., type=float, help="small constant to initialize W0 in SDELayer")
    parser.add_argument("--no_entire", action="store_false", help="whether to return entire trajectory. True if not specified.")
    parser.add_argument('--save_freq', type=int, default=10, help='frequency to save params and plots. Default 10.')
    parser.add_argument('--test_freq', type=int, default=10, help='frequency to eval on test set. Default 1.')
    parser.add_argument("--output", default="results/", type=str, help="output plots save directory. Default $cwd/results/.")

    hparams = parser.parse_args(args)

    return hparams


def main(hparams):
    lr = hparams.lr
    activ_fn = hparams.activn
    loss_fn = hparams.loss
    dataset = hparams.ds
    save_dir = hparams.output
    batch_size = hparams.batch_size
    settings = collections.defaultdict(bool)
    settings['driftw'] = hparams.no_drift
    settings['diffw'] = hparams.no_diff
    settings['kl'] = hparams.no_kl
    settings['diff_const'] = hparams.diff_const
    settings['prior_driftw'] = hparams.prior_dw
    settings['driftw_scale'] = hparams.driftw_scale
    settings['stop_grad'] = hparams.stop_grad
    settings['aug_dim'] = hparams.aug_dim
    settings['priorw0_sigma'] = hparams.priorw0_sigma
    settings['diff_drift'] = hparams.no_diff_drift
    settings['save_dir'] = hparams.output
    settings['compare_stl'] = False # don't train with the non-stl estimator
    epochs = hparams.epochs
    data_size = hparams.data_size
    key = random.PRNGKey(hparams.seed)
    num_plot_samples = hparams.num_samples
    hidden_dim = hparams.hidden_dim
    num_starts = hparams.num_starts
    entire = hparams.no_entire
    test_freq = hparams.test_freq
    save_freq = hparams.save_freq
    no_batching = hparams.no_batch
    kl_scale = 0 if not settings['kl'] else hparams.kl_scale 


    key, subkey = jax.random.split(key)
    inputs, targets, ds_test, get_batch, noise_std = get_data(dataset)(subkey, data_size, batch_size)
    assert data_size % batch_size == 0
    batch_per_epoch = data_size // batch_size

    output_dim = inputs.shape[-1]

    key, subkey = jax.random.split(key)
    (init_rand_params_fn, single_apply_fn), flat_w_dim = mlp(subkey, output_dim, hidden_dim, activ_fn, settings)

    @partial(jit, static_argnums=(3,))
    def predict_apply_fn(p, x, rng, entire):
        return single_apply_fn(p, x, rng=rng, entire=entire, settings=settings)

    key, subkey = jax.random.split(key)
    _, params = init_rand_params_fn(subkey, (-1, inputs.shape[-1]))

    opt_init, opt_update, get_params = optimizers.adam(lr)
    opt_state = opt_init(params)
    unravel_opt = ravel_pytree(opt_state)[1] 

    itr = 0
    loss = float('inf') # intial loss before opt update

    def callback(params, it, rng):
        print(f"Iteration: {it} elbo loss: {elbo(params, (inputs, targets), predict_apply_fn, rng=rng, kl_scale=kl_scale, loss_fn=loss_fn, noise_std=noise_std)}")

    @jit
    def update(i, opt_state, batch, rng, no_batch): 
        params = get_params(opt_state)
        loss_, gradient = value_and_grad(elbo)(params, batch, predict_apply_fn, rng, kl_scale, loss_fn, noise_std=noise_std) # scalar, (1, hidden_dim)
        return opt_update(i, gradient, opt_state), loss_

    def flatten(g):
        return ravel_pytree(g)[0]

    print("Starting training...")
    if no_batching:
        print("Using synthetic batch dataset")
        batch = inputs, targets

  # training loop
    for epoch in range(epochs):
        for i in range(batch_per_epoch):
            key, subkey = random.split(key)
            if not no_batching: # minibatch so that we don't update dynamics to squish, keep stable
                _, batch = get_batch(subkey, data_size, batch_size, output_dim)

        itr += 1
        if itr == 1 or itr % save_freq == 0:
            callback(params, itr, rng=subkey)
            if itr == 1:
                loss = elbo(params, batch, predict_apply_fn, key, kl_scale, loss_fn, noise_std=noise_std)

        if itr % test_freq == 0:
            test_loss_ = test_loss(params, ds_test, predict_apply_fn, loss_fn, key, noise_std=noise_std)
            mse_loss_ = test_loss(params, ds_test, predict_apply_fn, 'mse', key, noise_std=noise_std)
            loss_str = "iter: %d | test loss: %.6f" % (itr, test_loss_)
            print(loss_str)
        
        opt_state, loss = update(epoch, opt_state, batch, subkey, no_batching)
        params = get_params(opt_state)

if __name__ == '__main__':
    hparams = parse_args(sys.argv[1:])
    main(hparams)