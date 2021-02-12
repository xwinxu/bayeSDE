import argparse
import collections
import copy
import gc
import logging
import math
import os
import time

import torchbnn._impl.models as models
import torchbnn._impl.utils as utils
import tqdm

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter


def train(model, ema_model, optimizer, scheduler, epochs, global_step=0, output_dir=None, start_epoch=0,
          best_test=0, best_val=0, info=collections.defaultdict(dict), tb_writer=None):
    train_xent = utils.EMAMeter()
    train_accuracy = utils.EMAMeter()
    test_accuracy, best_test_acc = 0, best_test
    val_xent, val_xent_ema, val_accuracy, best_val_acc = 0, 0, 0, best_val
    obj, kl, ece, nfe = 0, 0, utils.AverageMeter(), utils.AverageMeter()
    epoch_start = time.time()
    for epoch in range(start_epoch, epochs):
        itr_per_epoch = 0
        for i, (x, y) in tqdm.tqdm(enumerate(train_loader)):
            model.train()
            model.zero_grad()
            x, y = x.to(device), y.to(device, non_blocking=True)
            logits, logqp = model(
                x, dt=args.dt, adjoint=args.adjoint, method=args.method, adaptive=args.adaptive, adjoint_adaptive=args.adjoint_adaptive, rtol=args.rtol, atol=args.atol
            )
            nfes = model.nfe
            xent = F.cross_entropy(logits, y, reduction="mean")
            loss = xent + args.kl_coeff * logqp
            obj, kl = loss, args.kl_coeff * logqp
            predictions = logits.detach().argmax(dim=1)
            accuracy = torch.eq(predictions, y).float().mean()
            train_ece = utils.score_model(logits.detach().cpu().numpy(), y.detach().cpu().numpy())[2]
            ece.step(train_ece)
            nfe.step(nfes)
            loss.backward()  # retain_graph=True
            optimizer.step()
            scheduler.step()
            train_xent.step(loss)
            train_accuracy.step(accuracy)
            utils.ema_update(model=model, ema_model=ema_model, gamma=args.gamma)
            global_step += 1
            itr_per_epoch += 1
            gc.collect()
            # per itr nfes: {global step: [train nfe, [for each pause_every:] val nfe, val nfe ema, test nfe,
            # test nfe ema]}
            info["nfes"] = {global_step: [nfes]}

            if global_step % args.pause_every == 0:
                # tb_writer.add_scalar(f'Grad Norm (pause/{args.pause_every})', torch.norm(x.grad), global_step)
                # TODO: magnitude of learned drift function
                # drift_y = model.f(0, aug_y)[:y.numel()]
                tb_writer.add_scalar(f'Activation Norm (pause/{args.pause_every})', torch.norm(y.detach().cpu().float()).numpy().mean(), global_step)
                tb_writer.add_scalar(f'NFE/train (pause/{args.pause_every})', nfes, global_step)
                val_xent, val_accuracy, val_ece, val_nfe = evaluate(model, validate=True)
                val_xent_ema, val_accuracy_ema, val_ece_ema, val_nfe_ema = evaluate(ema_model, validate=True)
                info['nfes'][global_step].extend([val_nfe, val_nfe_ema])
                if val_accuracy > best_val_acc:
                    best_val_acc = val_accuracy
                    utils.save_ckpt(model, ema_model, optimizer, os.path.join(output_dir, "best_val_acc.ckpt"),
                                    scheduler, epoch=epoch, global_step=global_step, best_acc=best_test_acc,
                                    best_val=best_val_acc, info=info)
                tb_writer.add_scalar('Accuracy/val', val_accuracy, global_step)
                tb_writer.add_scalar('Accuracy EMA/val', val_accuracy_ema, global_step)
                tb_writer.add_scalar('NLL/val', val_xent, global_step)
                tb_writer.add_scalar('NLL EMA/val', val_xent_ema, global_step)
                tb_writer.add_scalar('ECE/val', val_ece, global_step)
                tb_writer.add_scalar('ECE EMA/val', val_ece_ema, global_step)
                tb_writer.add_scalar('NFE/val (total/inference)', val_nfe, global_step)
                tb_writer.add_scalar('NFE EMA/val (total/inference)', val_nfe_ema, global_step)
                logging.warning(
                    f"global step: {global_step}, "
                    f"epoch: {epoch}, "
                    f"train_xent: {train_xent.val:.4f}, "
                    f"train_accuracy: {train_accuracy.val:.4f}, "
                    f"val_xent: {val_xent:.4f}, "
                    f"val_accuracy: {val_accuracy:.4f}, "
                    f"val_xent_ema: {val_xent_ema:.4f}, "
                    f"val_accuracy_ema: {val_accuracy_ema:.4f}"
                )

        epoch_time = epoch_start - time.time()
        utils.save_ckpt(model, ema_model, optimizer, os.path.join(output_dir, "state.ckpt"), scheduler, epoch=epoch,
                        global_step=global_step, best_val=best_val_acc, best_acc=best_test_acc, info=info)
        # import pdb; pdb.set_trace()
        tb_writer.add_scalar('Accuracy/train', train_accuracy.val, epoch)
        tb_writer.add_scalar('NLL/train', train_xent.val, epoch)
        tb_writer.add_scalar('KL/train', kl, epoch)
        tb_writer.add_scalar('Loss/train', obj, epoch)
        tb_writer.add_scalar('ECE/train', ece.val, epoch)
        tb_writer.add_scalar('NFE/train (avg/epoch)', nfe.val, epoch)
        nfe.__init__()  # reset for new epoch
        logging.warning("Wrote training scalars to tensorboard")

        test_xent, test_accuracy, test_ece, test_nfe = evaluate(model)
        test_xent_ema, test_accuracy_ema, test_ece_ema, test_nfe_ema = evaluate(ema_model)
        info['nfes'][global_step].extend([test_nfe, test_nfe_ema])
        tb_writer.add_scalar('Accuracy/test', test_accuracy, epoch)
        tb_writer.add_scalar('Accuracy EMA/test', test_accuracy_ema, epoch)
        tb_writer.add_scalar('NLL/test', test_xent, epoch)
        tb_writer.add_scalar('NLL EMA/test', test_xent_ema, global_step)
        tb_writer.add_scalar('ECE/test', test_ece, epoch)
        tb_writer.add_scalar('ECE EMA/test', test_ece_ema, epoch)
        tb_writer.add_scalar('NFE/test (total/inference)', test_nfe, epoch)
        tb_writer.add_scalar('NFE EMA/test (total/inference)', test_nfe_ema, epoch)
        logging.warning("Wrote test scalars to tensorboard")
        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            utils.save_ckpt(model, ema_model, optimizer, os.path.join(output_dir, "best_test_acc.ckpt"), scheduler,
                            epoch=epoch, global_step=global_step, best_val=best_val_acc, best_acc=best_test_acc,
                            info=info)
        with open(os.path.join(output_dir, "results.txt"), "a") as f:
            f.write(f"Epoch {epoch} (step {global_step}) in {epoch_time:.4f} sec | Train acc {train_accuracy.val}" + \
                    f" | Test accuracy {test_accuracy} | Test EMA accuracy {test_accuracy_ema}" + \
                    f" | Train NLL {train_xent.val} | Test NLL {test_xent} | Test EMA NLL {test_xent_ema} | Train "
                    f"Loss " + \
                    f" {obj.detach().cpu().numpy().tolist()} | Train KL {kl}" + \
                    f" | Train ECE {ece.val} | Test ECE {test_ece} | Test ECE EMA {test_ece_ema}" + \
                    f" | Train nfes {nfe.val} | Test NFE {test_nfe} | Test NFE EMA {test_nfe_ema}\n")
            logging.warning(f"Wrote epoch info to {os.path.join(output_dir, 'results.txt')}")
        info[global_step] = {'epoch': epoch, 'time': epoch_time, 'train_acc': train_accuracy.val,
                             'test_acc': test_accuracy,
                             'train_nll': train_xent.val, 'test_nll': test_xent, 'test_ema_nll': test_xent_ema,
                             'train_loss': obj.detach().cpu().numpy().tolist(),
                             'train_kl': kl.detach().cpu().numpy().tolist(), "val_acc": val_accuracy,
                             "val_xent": val_xent, "val_xent_ema": val_xent_ema,
                             "train_ece": ece.val, "test_ece": test_ece, "test_ece_ema": test_ece_ema,
                             "itr_per_epoch": itr_per_epoch, "avg_train_nfe": nfe.val,
                             "test_nfe": test_nfe, "test_nfe_ema": test_nfe_ema}
        utils.write_state_config(info, args.train_dir, file_name='state.json')


@torch.no_grad()
def _evaluate_with_loader(model, loader):
    xents = []
    accuracies = []
    eces = []
    nfes = 0
    model.eval()
    for i, (x, y) in enumerate(loader, 1):
        x, y = x.to(device), y.to(device, non_blocking=True)
        logits, _ = model(x, dt=args.dt, adjoint=args.adjoint, adjoint_adaptive=args.adjoint_adaptive, method=args.method)  # , rtol=args.rtol, atol=args.atol)
        loss = F.cross_entropy(logits, y, reduction="none")
        predictions = logits.detach().argmax(dim=1)
        accuracy = torch.eq(predictions, y).float()
        scores = utils.score_model(logits.detach().cpu().numpy(), y.detach().cpu().numpy())
        xents.append(loss)
        accuracies.append(accuracy)
        eces.append(torch.tensor([scores[2]]))
        nfes += model.nfe
        if i >= args.eval_batches: break
    return tuple(torch.cat(x, dim=0).mean(dim=0).cpu().item() for x in (xents, accuracies, eces)) + (nfes,)


def evaluate(model, validate=False):
    if validate:
        logging.warning("evaluating on validation set")
        test_xent, test_accuracy, test_ece, test_nfe = _evaluate_with_loader(model, val_loader)
    else:
        logging.warning("evaluating on test set")
        test_xent, test_accuracy, test_ece, test_nfe = _evaluate_with_loader(model, test_loader)
    return test_xent, test_accuracy, test_ece, test_nfe


def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_warmup_steps=0,
                                    num_cycles=7. / 16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda, last_epoch)


def main():
    input_chw = (3, 32, 32)
    if args.data == "mnist":
        input_chw = (1, 28, 28)
    if args.model == "baseline":
        model = models.BaselineYNet(
            input_size=input_chw,
            activation=args.activation,
            hidden_width=args.hidden_width
        )
    elif args.model == "sdebnn":
        model = models.SDENet(
            input_size=input_chw,
            inhomogeneous=args.inhomogeneous,
            activation=args.activation,
            verbose=args.verbose,
            hidden_width=args.hidden_width,
            weight_network_sizes=tuple(map(int, args.fw_width.split("-"))),
            blocks=tuple(map(int, args.nblocks.split("-"))),
            sigma=args.sigma,
            aug_dim=args.aug,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    ema_model = copy.deepcopy(model)
    model.to(device)
    ema_model.to(device)

    optimizer = optim.Adam(lr=args.lr, params=model.parameters())
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps=args.epochs * (50000 // args.batch_size))

    start_epoch, best_test_acc, best_val_acc, global_step = 0, 0, 0, 0
    info = collections.defaultdict(dict)
    if os.path.exists(os.path.join(args.train_dir, "state.ckpt")):
        # if os.path.exists(os.path.join(args.train_dir, "best_val_acc.ckpt")): # TODO: for debugging
        logging.warning("Loading checkpoints...")
        checkpoint = torch.load(os.path.join(args.train_dir, "state.ckpt"))
        # checkpoint = torch.load(os.path.join(args.train_dir, "best_val_acc.ckpt")) # TODO: for debugging
        start_epoch = checkpoint['epoch']
        best_test_acc = checkpoint['best_acc']
        best_val_acc = checkpoint['best_val_acc']
        info = checkpoint['info']
        global_step = checkpoint['global_step']
        model.load_state_dict(checkpoint["model"])
        ema_model.load_state_dict(checkpoint["ema_model"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        logging.warning(f"Successfully loaded checkpoints for epoch {start_epoch} | best acc {best_test_acc}")

    logging.warning(f'model: {model}')
    logging.warning(f'{utils.count_parameters(model) / 1e6:.4f} million parameters')

    tb_writer = SummaryWriter(os.path.join(args.train_dir, 'tb'))
    train(
        model, ema_model, optimizer, scheduler, args.epochs,
        output_dir=args.train_dir, global_step=global_step, start_epoch=start_epoch,
        best_test=best_test_acc, best_val=best_val_acc, info=info, tb_writer=tb_writer
    )
    tb_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=1000000)
    parser.add_argument('--no-gpu', action="store_true")
    parser.add_argument('--subset', type=int, default=None, help="Use subset of mnist data.")
    parser.add_argument('--data', type=str, default="cifar10", choices=['mnist', 'cifar10', 'cifar100'])
    parser.add_argument('--pin-memory', type=utils.str2bool, default=True)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--model', type=str, choices=['baseline', 'sdebnn'], default='sdebnn')
    parser.add_argument('--method', type=str, choices=['milstein', 'midpoint', "heun", "euler_heun"], default='midpoint')
    parser.add_argument('--gamma', type=float, default=0.999)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--aug', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--eval-batch-size', type=int, default=512)
    parser.add_argument('--pause-every', type=int, default=200)
    parser.add_argument('--eval-batches', type=int, default=10000)

    # Model.
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument('--atol', type=float, default=1e-4)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--adjoint', type=utils.str2bool, default=False)
    parser.add_argument('--adaptive', type=utils.str2bool, default=False)
    parser.add_argument('--adjoint_adaptive', type=utils.str2bool, default=False)
    parser.add_argument('--inhomogeneous', type=utils.str2bool, default=True)
    parser.add_argument('--activation', type=str, default="softplus",
                        choices=['swish', 'mish', 'softplus', 'tanh', 'relu', 'elu'])
    parser.add_argument('--verbose', type=utils.str2bool, default=False)
    parser.add_argument('--hidden-width', type=int, default=32)
    parser.add_argument('--fw-width', type=str, default="1-128-1")
    parser.add_argument('--nblocks', type=str, default="2-2-2")
    parser.add_argument('--sigma', type=float, default=0.1)

    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', type=utils.str2bool, default=True)
    parser.add_argument('--kl-coeff', type=float, default=1e-3, help='Coefficient on the KL term.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_gpu else 'cpu')
    torch.backends.cudnn.benchmark = True  # noqa

    utils.manual_seed(args)
    utils.write_config(args)

    print(args.pin_memory, args.num_workers)

    train_loader, val_loader, test_loader = utils.get_loader(
        args.data,
        train_batch_size=args.batch_size,
        test_batch_size=args.eval_batch_size,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        subset=args.subset,
        task="classification"
    )

    logging.warning(
        f"Training set size: {utils.count_examples(train_loader)}, "
        f"Val set size: {utils.count_examples(val_loader)}, "
        f"test set size: {utils.count_examples(test_loader)}"
    )

    main()
