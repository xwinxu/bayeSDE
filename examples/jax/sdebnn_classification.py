"""Train SDEBNN model on vision datasets. MNIST/CIFAR examples provided.
"""
import argparse
import itertools
import math
import os
import pickle
import time
from functools import partial

import numpy as np
import Path
from brax._impl import arch, brax
from brax._impl.resnet import resnet32v2
from brax.utils import utils
from brax.utils.datasets import get_dataset
from tqdm import tqdm

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.experimental import loops, optimizers, stax
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map, tree_multimap


def _nll(params, batch, rng):
    inputs, targets = batch
    preds, kl, info_dict = _predict(params, inputs, rng=rng, full_output=False)
    nll = -jnp.mean(jnp.sum(preds * targets, axis=1))
    return preds, nll, kl, info_dict


@partial(jit, static_argnums=(3,))
def sep_loss(params, batch, rng, kl_coef):  # no backprop
    preds, nll, kl, _ = _nll(params, batch, rng)
    if kl_coef > 0:
        obj_loss = nll + kl * kl_coef
    else:
        obj_loss = nll
    _sep_loss = {'loss': obj_loss, 'kl': kl, 'nll': nll, 'preds': preds}
    return obj_loss, _sep_loss


@partial(jit, static_argnums=(3,))
def loss(params, batch, rng, kl_coef):  # backprop so checkpoint
    _, nll, kl, _ = jax.checkpoint(_nll)(params, batch, rng)
    if kl_coef > 0:
        return nll + kl * kl_coef
    else:
        return nll

@jit
def predict(params, inputs, rng): 
    return _predict(params, inputs, rng=rng, full_output=True)

@partial(jit, static_argnums=(2,))
def accuracy(params, data, nsamples, rng):
    inputs, targets = data
    target_class = jnp.argmax(targets, axis=1)
    rngs = jax.random.split(rng, nsamples)
    preds, _, info_dic = vmap(predict, in_axes=(None, None, 0))(params, inputs, rngs)
    preds = jnp.stack(preds, axis=0)
    avg_preds = preds.mean(0)
    predicted_class = jnp.argmax(avg_preds, axis=1)
    n_correct = jnp.sum(predicted_class == target_class)
    n_total = inputs.shape[0]
    wts = info_dic['sdebnn_w']
    wts = jnp.stack(wts, axis=0)
    avg_wts = wts.mean(0)
    return n_correct, n_total, avg_preds, avg_wts

def update_ema(ema_params, params, momentum=0.999):
    return jax.tree_util.tree_multimap(lambda e, p: e * momentum + p * (1 - momentum), ema_params, params)


def evaluate(params, data_loader, input_size, nsamples, rng_generator, kl_coef):
    n_total = 0
    n_correct = 0
    nll = 0
    kl = 0
    logits = np.array([])
    wts = np.array([])
    labels = np.array([])
    for inputs, targets in data_loader:
        targets = jax.nn.one_hot(jnp.array(targets), num_classes=10)
        inputs = jnp.array(inputs).reshape((-1,) + (input_size[-1],) + input_size[:2])
        inputs = jnp.transpose(inputs, (0, 2, 3, 1))  # Permute from NCHW to NHWC
        batch_correct, batch_total, _logits, _wts = accuracy(
            params, (inputs, targets), nsamples, rng_generator.next()
        ) # _logits (nbatch, nclass)
        n_correct = n_correct + batch_correct
        _, batch_nll, batch_kl, _ = jit(_nll)(params, (inputs, targets), rng_generator.next())
        if n_total == 0:
            logits = np.array(_logits)
            wts = np.array(_wts)
            labels = np.array(targets)
        else:
            logits = np.concatenate([logits, np.array(_logits)], axis=0)
            wts = np.concatenate([wts, np.array(_wts)], axis=0)
            labels = np.concatenate([labels, targets], axis=0)
        n_total = n_total + batch_total
        nll = nll + batch_nll
        kl = kl + batch_kl
    return n_correct / n_total, jnp.stack(logits, axis=0), labels, nll / n_total, kl / n_total, jnp.stack(wts, axis=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SDE-BNN CIFAR10 Training")
    parser.add_argument("--model", type=str, choices=["resnet", "sdenet"], default="sdenet")
    parser.add_argument("--output", type=str, default="./output", help="(default: %(default)s)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stl", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3, help="(default: %(default)s)")
    parser.add_argument("--epochs", type=int, default=500, help="(default: %(default)s)")
    parser.add_argument("--bs", type=int, default=128, help="(default: %(default)s)")
    parser.add_argument("--test_bs", type=int, default=1000)
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--w_init", type=float, default=-1., help="scale the last layer W init of w_net. default: $(default)")
    parser.add_argument("--b_init", type=float, default=-1., help="scale the last layer b init of w_net. default: $(default)")
    parser.add_argument("--p_init", type=float, default=-1., help="scale the output of w_net by exp(p) so that diffusion doesn't overpower. default: $(default)")

    parser.add_argument("--no_drift", action="store_true")
    parser.add_argument("--ou_dw", action="store_false", help="OU prior on dw (difference parameterization)")
    parser.add_argument("--kl_coef", type=float, default=1e-3, help="(default: %(default)s)")
    parser.add_argument("--diff_coef", type=float, default=1e-4, help="(default: %(default)s)")
    parser.add_argument("--ds", type=str, choices=["mnist", "cifar10"], default="cifar10", help="(default: %(default)s)")
    parser.add_argument("--no_xt", action="store_false", help="time dependent")
    parser.add_argument("--acc_grad", type=int, default=1)
    parser.add_argument("--aug", type=int, default=0, help="(default: %(default)s)")
    parser.add_argument("--remat", action="store_true")
    parser.add_argument("--ema", type=float, default=0.999)
    parser.add_argument("--meanfield_sdebnn", action="store_true")

    parser.add_argument("--disable_test", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--nblocks", type=str, default="1", help="dash-separated integers (default: %(default)s)")
    parser.add_argument("--nsteps", type=int, default=20, help="(default: %(default)s)")
    parser.add_argument("--block_type", type=int, choices=[0, 1, 2], default=0, help="(default: %(default)s)")
    parser.add_argument("--fx_dim", type=int, default=32, help="(default: %(default)s)")
    parser.add_argument("--fx_actfn", type=str, choices=["softplus", "tanh", "elu", "swish", "rbf"], default="softplus", help="(default: %(default)s)")
    parser.add_argument("--fw_dims", type=str, default="1-64-1", help="dash-separated integers (default: %(default)s)")
    parser.add_argument("--fw_actfn", type=str, choices=["softplus", "tanh", "elu", "swish", "rbf"], default="softplus", help="(default: %(default)s)")
    parser.add_argument("--lr_sched", type=str, choices=['constant', 'custom', 'custom2', 'stair', 'exp', 'inv', 'cos', 'warmup'], default="constant", help="(default: %(default)s)")
    args = parser.parse_args()
    print(args)

    rng_generator = utils.jaxRNG(seed=args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    params_path = os.path.join(str(output_dir), "params.pkl")

    train_loader, train_eval_loader, val_loader, test_loader, input_size, train_size = get_dataset(args.bs, args.test_bs, args.ds)
    num_batches = len(train_loader)
    print(f"Number of batches: {num_batches}")
    train_batches = utils.inf_generator(train_loader)

    mf = partial(brax.MeanField, disable=True) if args.kl_coef == 0. else brax.MeanField

    if args.model == "resnet":
        layers = resnet32v2(expansion=1,
                            normalization_method="",
                            use_fixup=False,
                            option="B",
                            init="he",
                            actfn=stax.Relu)
        resnet = stax.serial(*layers, stax.Flatten, stax.Dense(10), stax.LogSoftmax)
        init_random_params, _predict = brax.bnn_serial(mf(resnet))
    else:
        # SDE BNN.
        fw_dims = list(map(int, args.fw_dims.split("-")))

        layers = [mf(arch.Augment(args.aug))]
        nblocks = list(map(int, args.nblocks.split("-")))
        for i, nb in enumerate(nblocks):
            fw = arch.MLP(fw_dims, actfn=args.fw_actfn, xt=args.no_xt, ou_dw=args.ou_dw, nonzero_w=args.w_init, nonzero_b=args.b_init, p_scale=args.p_init)  # weight network is time dependent
            if args.meanfield_sdebnn:
                layers.extend([mf(brax.SDEBNN(args.block_type,
                                              args.fx_dim,
                                              args.fx_actfn,
                                              fw,
                                              diff_coef=args.diff_coef,
                                              stl=args.stl,
                                              xt=args.no_xt,
                                              nsteps=args.nsteps,
                                              remat=args.remat,
                                              w_drift=not args.no_drift,
                                              stax_api=True))
                               for _ in range(nb)])
            else:
                layers.extend([brax.SDEBNN(args.block_type,
                                           args.fx_dim,
                                           args.fx_actfn,
                                           fw,
                                           diff_coef=args.diff_coef,
                                           stl=args.stl,
                                           xt=args.no_xt,
                                           nsteps=args.nsteps,
                                           remat=args.remat,
                                           w_drift=not args.no_drift)
                               for _ in range(nb)])
            if i < len(nblocks) - 1:
                layers.append(mf(arch.SqueezeDownsample(2)))
        layers.append(mf(stax.serial(stax.Flatten, stax.Dense(10), stax.LogSoftmax)))

        init_random_params, _predict = brax.bnn_serial(*layers)

    lr = args.lr if args.lr_sched == "constant" else utils.get_lr_schedule(args.lr_sched, num_batches, args.lr)
    opt_init, opt_update, get_params = optimizers.adam(lr)

    loss_grad_fn = jit(vmap(grad(loss), in_axes=(None, None, 0, None)), static_argnums=(3,))

    def update(i, opt_state, batch, acc_grad, nsamples, base_rng):
        params = get_params(opt_state)
        bsz = int(math.ceil(batch[0].shape[0] / acc_grad))
        first_batch = (batch[0][:bsz], batch[1][:bsz])

        rngs = jax.random.split(base_rng, nsamples)
        grads = loss_grad_fn(params, first_batch, rngs, args.kl_coef / train_size)

        grad_std = tree_map(lambda bg: jnp.std(bg, 0), grads)
        avg_std = jnp.nanmean(ravel_pytree(grad_std)[0])

        grads = tree_map(lambda bg: jnp.mean(bg, 0), grads)

        grad_snr = tree_multimap(lambda m, sd: jnp.abs(m / sd), grads, grad_std)
        avg_snr = jnp.nanmean(ravel_pytree(grad_snr)[0])

        for i in range(1, acc_grad):
            batch_i = (batch[0][(i - 1) * bsz:i * bsz], batch[1][(i - 1) * bsz:i * bsz])
            grads_i = loss_grad_fn(params, batch_i, rngs, args.kl_coef / train_size)
            grads_i = tree_map(lambda bg: jnp.mean(bg, 0), grads_i)
            grads = tree_multimap(lambda g, g_new: (g * i + g_new) / (i + 1), grads, grads_i)

        pre_update = get_params(opt_state)
        post_update = jit(opt_update)(i, grads, opt_state)
        assert jnp.not_equal(ravel_pytree(pre_update)[0], ravel_pytree(get_params(post_update))[0]).any()
        return post_update, avg_std, avg_snr

    out_shape, init_params = init_random_params(rng_generator.next(), (-1,) + input_size)
    opt_state = opt_init(init_params)
    unravel_opt = ravel_pytree(opt_state)[1] 
    ema_params = init_params
    best_val_acc, best_test_acc = 0.0, 0.0
    flat_params, unravel_params = ravel_pytree(get_params(opt_state))  # for pickling params

    itercount = itertools.count()
    print("\nStarting training...", flush=True)
    for epoch in range(args.epochs):
        print(f"Epoch {epoch} | Iteration: {itr}")
        start_time = time.time()
        gradstd_meter = utils.AverageMeter()
        gradsnr_meter = utils.AverageMeter()
        for i in tqdm(range(num_batches)):
            inputs, targets = next(train_batches)
            targets = jax.nn.one_hot(jnp.array(targets), num_classes=10)
            inputs = jnp.array(inputs).reshape((-1,) + (input_size[-1],) + input_size[:2])
            # Permute from NCHW (Pytorch) to NHWC (JAX)
            inputs = jnp.transpose(inputs, (0, 2, 3, 1))
            opt_state, gradstd, grad_snr = update(next(itercount), opt_state, (inputs, targets), args.acc_grad, 1, rng_generator.next()) # train on 1 sample

            gradstd_meter.update(float(gradstd), n=inputs.shape[0])
            gradsnr_meter.update(float(grad_snr), n=inputs.shape[0])

            ema_params = update_ema(ema_params, get_params(opt_state), momentum=args.ema)
            itr = int(repr(itercount)[6:-1])
        epoch_time = time.time() - start_time
        epoch_info = sep_loss(get_params(opt_state), (inputs, targets), rng_generator.next(), args.kl_coef / train_size)[-1]
        # check nan
        no_nans = utils.check_nans([epoch_info['loss'], epoch_info['nll']])
        if not no_nans:
            with open(os.path.join(str(output_dir), f"{epoch}_nan.pkl"), 'wb') as f:
                pickle.dump(get_params(opt_state), f)
                exit(f"nan encountered in epoch_info and params pickled: {epoch_info}")

        params = get_params(opt_state)
        train_acc, train_logits, train_labels, train_nll, train_kl, _ = evaluate(params, train_eval_loader, input_size, args.nsamples, rng_generator, args.kl_coef / train_size)
        train_loss = train_nll + args.kl_coef * train_kl
        if args.disable_test:
            val_acc, val_logits, val_labels, val_nll, val_kl = jnp.zeros(1), jnp.zeros_like(train_logits), jnp.zeros(1), jnp.zeros(1)
            test_acc, test_logits, test_labels, test_nll, test_kl = jnp.zeros(1), jnp.zeros_like(train_logits), jnp.zeros(1), jnp.zeros(1)
        else:
            val_acc, val_logits, val_labels, val_nll, val_kl, _ = evaluate(ema_params, val_loader, input_size, args.nsamples, rng_generator, args.kl_coef / train_size)
            test_acc, test_logits, test_labels, test_nll, test_kl, test_ws = evaluate(ema_params, test_loader, input_size, args.nsamples, rng_generator, args.kl_coef / train_size)
        val_loss, test_loss = val_nll + args.kl_coef * val_kl, test_nll + args.kl_coef * test_kl

        cal_train = utils.get_calibration(train_labels, jax.device_get(jnp.exp(train_logits)))
        cal_val= utils.get_calibration(val_labels, jax.device_get(jnp.exp(val_logits)))
        cal_test = utils.get_calibration(test_labels, jax.device_get(jnp.exp(test_logits)))
        score_train = utils.score_model(np.asarray(jnp.exp(train_logits)), train_labels, bins=10)
        score_val = utils.score_model(np.asarray(jnp.exp(val_logits)), val_labels, bins=10)
        score_test = utils.score_model(np.asarray(jnp.exp(test_logits)), test_labels, bins=10)

        if args.verbose:
            print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
            print("Training set accuracy {}".format(train_acc))
            print("Val set accuracy {}".format(val_acc))
            print("Test set accuracy {}".format(test_acc))
            print("Training set loss {}".format(train_loss))

            print("Grad STD {}".format(gradstd_meter.avg))
            print("Grad SNR {}".format(gradsnr_meter.avg), flush=True)
            print(f"Train scores: {score_train} | cal ECE: {cal_train['ece']}")
            print(f"Val scores: {score_val} | cal ECE: {cal_val['ece']}")
            print(f"Test scores: {score_test} | cal ECE: {cal_test['ece']}")

            print(f"shapes: {train_logits.shape} | {val_logits.shape} | {test_logits.shape}")

        if best_val_acc < val_acc:
            best_val_acc = val_acc
            print("Best Val Acc", best_val_acc)

        if best_test_acc < test_acc:
            best_test_acc = test_acc
            print("Best Test Acc", best_test_acc)

    print(f"Finished successfully {args.epochs} epochs :)")
