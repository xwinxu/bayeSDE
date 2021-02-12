"""All-purpose utility file with helpers for

- PyTorch tensor processing (flattening, unflattening, vjp, jvp),
- plotting,
- custom gradient checking (gradient wrt parameters, and second-order gradients),
- meters (ema, online average),
- ema model averaging,
- google cloud storage utilities,
- custom context managers (Timer, DisableGC),
- checkpoint storage/loading, and
- data loaders.
"""
import abc
import argparse
import collections
import copy
import datetime
import gc
from genericpath import exists
import io
import json
import logging
import math
import os
import random
import shutil
import sys
import time
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import six
import torch
import torch.nn.functional as F
import torchvision as tv
import tqdm
from scipy import stats
from torch import nn, optim
from torch.utils import data


# Misc.
def alleq(l: Sequence, f: Optional[Callable] = lambda x, y: x == y):
    """Check all arguments in a sequence are equal according to a given criterion.
    Args:
        f: A bi-variate boolean function.
        l: A list/tuple.
    Returns:
        True if everything is equal; otherwise False.
    """
    return all(f(l[0], li) for li in l[1:])


def zip_(*args: Sequence):
    """Assert sequences of same length before zipping."""
    if len(args) == 0: return []
    assert alleq(args, lambda x, y: len(x) == len(y))
    return zip(*args)

def write_state_config(state_dict: dict, train_dir: str, file_name='state.json'):
    os.makedirs(train_dir, exist_ok=True)
    config_path = os.path.join(train_dir, file_name)
    with open(config_path, 'w') as f:
        json.dump(state_dict, f, indent=4)
    logging.warning(f"Wrote state config: {config_path}")

def write_config(args: argparse.Namespace, file_name='config.json'):
    os.makedirs(args.train_dir, exist_ok=True)
    os.makedirs(os.path.join(args.train_dir, 'ckpts'), exist_ok=True)
    config_path = os.path.join(args.train_dir, file_name)
    with open(config_path, 'w') as f:
        json.dump(args.__dict__, f, indent=4)
    logging.warning(f"Wrote config: {config_path}")

    if hasattr(args, 'cloud_storage') and args.cloud_storage:
        gs_upload_from_path(config_path)
        logging.warning(f"Uploaded to cloud: {config_path}")


def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def str2int(v):
    if isinstance(v, int): return v
    if v.lower() in ("none",): return None
    return int(v)


def gather_args(parser: argparse.ArgumentParser):
    """Gathers known and unknown args together.

    Unknown args are arguments whose names we don't known before hand, and they aren't specified by `add_argument`.
    """
    args, unknown_args = parser.parse_known_args()
    unknown_options = collections.defaultdict(list)

    key = None
    for arg in unknown_args:
        if arg.startswith('--'):
            key = arg[2:]
        elif arg.startswith('-'):
            key = arg[1:]
        else:
            unknown_options[key].append(arg)
    args.__dict__ = {**args.__dict__, **unknown_options}
    return args


def flatten_nested_pystruct(sequence: Sequence):
    """Flatten nested python list/tuple/set and return a list of elements."""
    if not isinstance(sequence, (tuple, list, set)):
        return [sequence]
    return [i for entry in sequence for i in flatten_nested_pystruct(entry)]


def parallel_sort(*args, key=None, reverse=False):
    """Parallel sort of multiple lists."""
    # args: A bunch of sequences.
    if key is None: key = lambda inputs: inputs[0]  # Parallel sort based on the order of the first list.
    ret = sorted(zip_(*args), key=key, reverse=reverse)
    return tuple([ret_i[j] for ret_i in ret] for j in range(len(args)))


def linregress_slope(x, y):
    """Return the slope of a least-squares regression for two sets of measurements."""
    return stats.linregress(x, y)[0]


def pretty_str(names: Sequence, vars: Sequence, precision: Optional[float] = 4):
    ret = ""
    for name, var in zip(names[:-1], vars[:-1]):
        if isinstance(var, float):
            ret += f"{name}: {var:.{precision}f}, "
        else:
            ret += f"{name}: {var}, "
    ret += f"{names[-1]}: {vars[-1]}"  # No comma after last.
    return ret


def mytqdm(it, cloud_storage, *argv, **kwargs):
    if cloud_storage: return it
    return tqdm.tqdm(it)


def show_env(args_or_device):
    if hasattr(args_or_device, "device"):
        args_or_device = args_or_device.device
    logging.warning(f"Running on device: {args_or_device}")
    logging.warning(f"Running Python: \n{sys.version}; \nversion info: \n{sys.version_info}")
    logging.warning(f"Running PyTorch: {torch.__version__}")
    logging.warning(f"Running six: {six.__version__}")


# Torch.
def flatten(possibly_sequence: Union[torch.Tensor, Sequence[torch.Tensor]]):
    if torch.is_tensor(possibly_sequence): return possibly_sequence.reshape(-1)
    return torch.cat([p.reshape(-1) for p in possibly_sequence]) if len(possibly_sequence) > 0 else torch.tensor([])


def flatten_nested(possibly_sequence: Union[torch.Tensor, Sequence]):
    if torch.is_tensor(possibly_sequence): return possibly_sequence.reshape(-1)
    flat_tensors = [flatten_nested(entry) for entry in possibly_sequence]
    return torch.cat(flat_tensors, dim=0) if len(flat_tensors) > 0 else torch.tensor([])


def ravel_pytree(possibly_sequence: Union[Sequence, torch.Tensor]) -> Tuple[torch.Tensor, Callable]:
    if torch.is_tensor(possibly_sequence):
        return possibly_sequence.reshape(-1), lambda x: x.reshape(possibly_sequence.size())

    def make_unravel(size):  # Need this function to copy size!
        return lambda x: x.reshape(size)

    unravels, flats, numels = [], [], []
    for entry in possibly_sequence:
        if torch.is_tensor(entry):
            unravel_i = make_unravel(entry.size())
            flat_i = entry.reshape(-1)
        else:
            flat_i, unravel_i = ravel_pytree(entry)
        unravels.append(unravel_i)
        flats.append(flat_i)
        numels.append(flat_i.numel())

    def unravel(flat: torch.Tensor):
        return [unravel_(flat_) for flat_, unravel_ in zip_(flat.split(split_size=numels), unravels)]

    return torch.cat(flats) if len(flats) > 0 else torch.tensor([]), unravel


def fill_tail_dims(y: torch.Tensor, y_like: torch.Tensor):
    """Fill in missing trailing dimensions for y according to y_like."""
    return y[(...,) + (None,) * (y_like.dim() - y.dim())]


def channel_cat(t, y):
    t = fill_tail_dims(t, y).expand_as(y[:, :1, ...])
    return torch.cat((t, y), dim=1)


class Swish(nn.Module):
    def __init__(self, beta=.5):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x):
        return _swish(x, beta=self.beta)

    # Just to make pylint happy...
    def _forward_unimplemented(self, *input: Any) -> None:
        pass


@torch.jit.script
def _swish(x, beta):
    return x * torch.sigmoid(x * F.softplus(beta))


class Mish(nn.Module):
    def forward(self, x):
        return _mish(x)


@torch.jit.script
def _mish(x):
    return x * torch.tanh(F.softplus(x))


def flat_to_shape(tensor: torch.Tensor, shapes: Sequence[Union[torch.Size, Sequence]], length=()):
    tensor_list = []
    total = 0
    for shape in shapes:
        next_total = total + shape.numel()  # noqa
        # It's important that this be view((...)), not view(...). Else when length=(), shape=() it fails.
        tensor_list.append(tensor[..., total:next_total].view((*length, *shape)))
        total = next_total
    return tensor_list


def convert_none_to_zeros(sequence: Sequence[Union[torch.Tensor, type(None)]], like_sequence: Sequence[torch.Tensor]):
    return [torch.zeros_like(q) if p is None else p for p, q in zip(sequence, like_sequence)]


def make_seq_requires_grad(sequence: Sequence[torch.Tensor]):
    return [p if p.requires_grad else p.detach().requires_grad_(True) for p in sequence]


def is_strictly_increasing(ts):
    return all(x < y for x, y in zip(ts[:-1], ts[1:]))


def is_nan(t):
    return torch.any(torch.isnan(t))


def vjp(outputs, inputs, **kwargs):
    if torch.is_tensor(inputs):
        inputs = [inputs]
    _dummy_inputs = [torch.as_strided(i, (), ()) for i in inputs]  # Workaround for PyTorch bug #39784.

    if torch.is_tensor(outputs):
        outputs = [outputs]
    outputs = make_seq_requires_grad(outputs)

    _vjp = torch.autograd.grad(outputs, inputs, **kwargs)
    return convert_none_to_zeros(_vjp, inputs)


def jvp(outputs, inputs, grad_inputs=None, **kwargs):
    if torch.is_tensor(inputs):
        inputs = [inputs]
    _dummy_inputs = [torch.as_strided(i, (), ()) for i in inputs]  # Workaround for PyTorch bug #39784.

    if torch.is_tensor(outputs):
        outputs = [outputs]
    outputs = make_seq_requires_grad(outputs)

    dummy_outputs = [torch.zeros_like(o, requires_grad=True) for o in outputs]
    _vjp = torch.autograd.grad(outputs, inputs, grad_outputs=dummy_outputs, **kwargs)
    _jvp = torch.autograd.grad(_vjp, dummy_outputs, grad_outputs=grad_inputs, **kwargs)
    return convert_none_to_zeros(_jvp, dummy_outputs)


def to_numpy(*possibly_tensors: Union[torch.Tensor, np.ndarray]):
    arrays = [t.detach().cpu().numpy() if torch.is_tensor(t) else t for t in possibly_tensors]
    return arrays[0] if len(arrays) == 1 else arrays


def manual_seed(seed_or_args: Union[int, argparse.Namespace]):
    if hasattr(seed_or_args, 'seed'):
        seed_or_args = seed_or_args.seed
    random.seed(seed_or_args)
    np.random.seed(seed_or_args)
    torch.manual_seed(seed_or_args)


def manual_dtype(dtype_or_args: Union[str, argparse.Namespace]):
    dtype = dtype_or_args.dtype if hasattr(dtype_or_args, 'dtype') else dtype_or_args
    if dtype in ('float64', 'double'):
        torch.set_default_dtype(torch.float64)
    elif dtype in ('float16', 'half'):
        torch.set_default_dtype(torch.float16)


def count_parameters(*modules: nn.Module, only_differentiable: Optional[bool] = False):
    """Count the number of parameters for each module."""
    param_lists = [list(m.parameters()) for m in modules]
    if only_differentiable:
        param_lists = [[p for p in param_list if p.requires_grad] for param_list in param_lists]
    numels = [sum(p.numel() for p in param_list) for param_list in param_lists]
    return numels[0] if len(modules) == 1 else numels


def count_parameter_tensors(*modules: nn.Module, only_differentiable: Optional[bool] = False):
    param_lists = [list(m.parameters()) for m in modules]
    if only_differentiable:
        param_lists = [[p for p in param_list if p.requires_grad] for param_list in param_lists]
    lens = [len(param_list) for param_list in param_lists]
    return lens[0] if len(modules) == 1 else lens


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = tuple(shape)

    def forward(self, x):
        return x.view((-1,) + self.shape)


def subsequent_mask(size, device=None):
    """Mask out subsequent positions.

    Useful for transformer training.
    """
    return torch.triu(torch.ones((1, size, size), device=device), diagonal=1) == 0


def masks_from_lengths(lengths: torch.Tensor):
    """Create True/False mask based on lengths.

    Useful for masking out padding tokens.
    """
    return torch.arange(max(lengths), device=lengths.device)[None, :] < lengths[:, None]


def evaluate_prettiness(sampler=None,
                        folder=None,
                        input_2='cifar10-train',
                        n=50000,
                        batch_size=1000,
                        clean_afterwards=False,
                        fid=False,
                        isc=False,
                        kid=False):
    """Evaluate a generative model in terms of IS, FID, or KID.

    At least one of `model` or `folder` must be present.

    Args:
        sampler (object, optional): An objective with the method `func` that samples from the model.
        folder (str, optional): Path to the folder that contains all the images.
        input_2 (str, optional): Name of registered dataset or a path to a folder.
        n (int, optional): Number of samples to take.
        batch_size (int, optional): Number of samples in each batch.
        clean_afterwards (bool, optional): Clean the local cache if True.

    Returns:
        A dictionary of metric values.
    """
    import torch_fidelity

    if sampler is None and folder is None:
        raise ValueError(f"model and folder cannot both be none")

    if folder is None:
        now = datetime.datetime.now().strftime("%d:%m:%Y-%H:%M:%S")
        folder = os.path.join(os.path.expanduser("~"), 'evaluate_prettiness', f'{now}')
        os.makedirs(folder, exist_ok=True)

        idx = 0
        for _ in tqdm.tqdm(range(n // batch_size), desc='spawn samples'):
            batch = sampler(batch_size=batch_size).detach().cpu().numpy()
            if batch.shape[1] == 3:
                batch = batch.transpose((0, 2, 3, 1))
            for img in batch:
                img_path = os.path.join(folder, f'{idx:06d}.png')
                plt.imsave(img_path, img)
                idx += 1

    stats = torch_fidelity.calculate_metrics(folder, input_2, isc=isc, fid=fid, kid=kid)
    if clean_afterwards: shutil.rmtree(folder)
    return stats


# Adapted from https://github.com/pytorch/examples/blob/master/word_language_model/model.py
# `batch_first` is a new argument; this argument has been tested.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.get_default_dtype()).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        if not batch_first:
            pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)
        self.batch_first = batch_first

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        offset = self.pe[:, :x.size(1)] if self.batch_first else self.pe[:x.size(0), :]
        x = x + offset
        return self.dropout(x)


def inspect_tensor(t: torch.Tensor, name=''):
    t = t.view(-1)
    msg = f'{name} min: {t.min()}, max: {t.max()}, has nan: {torch.isnan(t).any()}'
    logging.warning(msg)
    return msg


def inspect_module(module, name=''):
    flat_params = [p.flatten() for p in module.parameters()]
    if len(flat_params) > 0:
        flat_params = torch.cat(flat_params)
        logging.warning(
            f'{name} param, '
            f'max abs: {flat_params.abs().max():.4f}, min abs: {flat_params.abs().min():.4f}, '
            f'has nan: {torch.isnan(flat_params).any()}'
        )
    else:
        logging.warning(f'module {name} no param')

    flat_grads = [p.grad.flatten() for p in module.parameters() if p.grad is not None]
    if len(flat_grads) > 0:
        flat_grads = torch.cat(flat_grads)
        logging.warning(
            f'{name} grad, '
            f'max abs: {flat_grads.abs().max():.4f}, min abs: {flat_grads.abs().min():.4f}, '
            f'has nan: {torch.isnan(flat_params).any()}'
        )
    else:
        logging.warning(f'module {name} no grad')


class LinearLR(object):

    def __init__(self, optimizer, init_lr, term_lr, last_epoch):
        super(LinearLR, self).__init__()
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.term_lr = term_lr
        self.last_epoch = last_epoch

        self._i = 0

    def step(self):
        self._i += 1
        i = min(self._i, self.last_epoch)
        lr = self.init_lr + (self.term_lr - self.init_lr) * i / self.last_epoch
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class OptimizedModel(abc.ABC, nn.Module):

    # Slightly faster than the `zero_grad` from library.
    def zero_grad(self) -> None:
        for p in self.parameters(): p.grad = None


class VerboseSequential(OptimizedModel):

    def __init__(self, *args, verbose=False, stream: str = 'stdout'):
        super(VerboseSequential, self).__init__()
        self.layers = nn.ModuleList(args)
        self.forward = self._forward_verbose if verbose else self._forward
        self.stream = stream  # Don't use the stream from `sys`, since we can't serialize them!

    def _forward_verbose(self, net):
        stream = (
            {'stdout': sys.stdout, 'stderr': sys.stderr}[self.stream]
            if self.stream in ('stdout', 'stderr') else self.stream
        )
        print(f'Input size: {net.size()}', file=stream)
        for i, layer in enumerate(self.layers):
            net = layer(net)
            print(f'Layer {i}, output size: {net.size()}', file=stream)
        return net

    def _forward(self, net):
        for layer in self.layers:
            net = layer(net)
        return net


# Plotting.
def plot(img_path=None, plots=(), scatters=(), hists=(), errorbars=(), options=None):
    """A multi-functional plotter; reduces boilerplate.

    Good features of this plotter are:
        1): Tweaked dpi.
        2): Enabled tight_layout.
        3): Plot closing.

    Args:
        img_path (str): A path to the place where the image should be written.
        plots (list of dict, optional): A list of curves that needs `plt.plot`.
        scatters (list of dict, optional): A list of scatter plots that needs `plt.scatter`.
        hists (list of histograms, optional): A list of histograms that needs `plt.hist`.
        errorbars (list of errorbars, optional): A list of errorbars that needs `plt.errorbar`.
        options (dict, optional): A dictionary of optional arguments, such as title, xlabel, ylabel, etc.

    Returns:
        Nothing.
    """
    if options is None: options = {}

    plt.figure(dpi=300)
    if 'xscale' in options: plt.xscale(options['xscale'])
    if 'yscale' in options: plt.yscale(options['yscale'])
    if 'xlabel' in options: plt.xlabel(options['xlabel'])
    if 'ylabel' in options: plt.ylabel(options['ylabel'])
    if 'title' in options: plt.title(options['title'])

    for entry in plots:
        kwargs = {key: entry[key] for key in entry if key != 'x' and key != 'y'}
        plt.plot(entry['x'], entry['y'], **kwargs)
    for entry in errorbars:
        kwargs = {key: entry[key] for key in entry if key != 'x' and key != 'y'}
        plt.errorbar(entry['x'], entry['y'], **kwargs)
    for entry in scatters:
        kwargs = {key: entry[key] for key in entry if key != 'x' and key != 'y'}
        plt.scatter(entry['x'], entry['y'], **kwargs)
    for entry in hists:
        kwargs = {key: entry[key] for key in entry if key != 'x'}
        plt.hist(entry['x'], **kwargs)

    if len(plots) > 0 or len(scatters) > 0 or len(errorbars) > 0: plt.legend()
    plt.tight_layout()
    plt.savefig(img_path)
    plt.close()


def plot_side_by_side(figs1,
                      figs2,
                      nrows=8,
                      ncols=1,
                      img_path=None,
                      dpi=300,
                      title=None,
                      left_title=None,
                      right_title=None,
                      frameon=True,
                      max_batch_size=64):
    """Plot a dictionary of figures.

    Parameters
    ----------
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """
    os.makedirs(os.path.dirname(img_path), exist_ok=True)

    figs1, figs2 = figs1.squeeze(), figs2.squeeze()
    if isinstance(figs1, torch.Tensor):
        figs1 = to_numpy(figs1)

    if isinstance(figs2, torch.Tensor):
        figs2 = to_numpy(figs2)

    assert figs1.shape == figs2.shape
    figs1, figs2 = figs1[:max_batch_size, ...], figs2[:max_batch_size, ...]

    if nrows * ncols < len(figs1):
        ncols = (len(figs1) + nrows - 1) // nrows
    assert nrows * ncols >= len(figs1)

    fig = plt.figure(dpi=dpi, frameon=frameon)
    outer = gridspec.GridSpec(1, 2, wspace=0.05, hspace=0.05)

    if left_title is not None:
        ax = plt.Subplot(fig, outer[0])
        ax.set_title(left_title)
        ax.axis('off')
        fig.add_subplot(ax)

    left_block = gridspec.GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=outer[0], wspace=0.0, hspace=0.0)
    for ind, item in enumerate(figs1):
        ax = plt.Subplot(fig, left_block[ind])
        ax.set_axis_off()
        ax.set_aspect('auto')

        if isinstance(figs1, dict):
            # `item` is the key.
            img = figs1[item]
            cmap = plt.gray() if len(img.shape) == 2 else None
            ax.imshow(img, cmap=cmap)
            ax.set_title(item)
        else:
            # `item` is the image.
            cmap = plt.gray() if len(item.shape) == 2 else None
            item = item.transpose(1, 2, 0) if item.shape[0] in (1, 3) else item
            ax.imshow(item, cmap=cmap)
        fig.add_subplot(ax)

    if right_title is not None:
        ax = plt.Subplot(fig, outer[1])
        ax.set_title(right_title)
        ax.axis('off')
        fig.add_subplot(ax)

    right_block = gridspec.GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=outer[1], wspace=0.0, hspace=0.0)
    for ind, item in enumerate(figs2):
        ax = plt.Subplot(fig, right_block[ind])
        ax.set_axis_off()
        ax.set_aspect('auto')

        if isinstance(figs2, dict):
            # `item` is the key.
            img = figs2[item]
            cmap = plt.gray() if len(img.shape) == 2 else None
            ax.imshow(img, cmap=cmap)
            ax.set_title(item)
        else:
            # `item` is the image.
            cmap = plt.gray() if len(item.shape) == 2 else None
            item = item.transpose(1, 2, 0) if item.shape[0] in (1, 3) else item
            ax.imshow(item, cmap=cmap)
        fig.add_subplot(ax)

    fig.suptitle(title)
    plt.savefig(img_path, bbox_inches='tight')
    plt.clf()
    plt.close()


def make_mp4(img_paths, out_path: str, fps):
    import cv2  # Don't import unless absolutely necessary!
    if not out_path.endswith(".mp4"): raise ValueError(f"`out_path` must specify path to .mp4 file type")
    frame = cv2.imread(img_paths[0])
    cv2.imshow('video', frame)
    height, width, channels = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    for img_path in img_paths:
        frame = cv2.imread(img_path)
        out.write(frame)
        cv2.imshow('video', frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit `q` to exit
            break

    out.release()
    cv2.destroyAllWindows()


# Gradient checking.
def swiss_knife_gradcheck(func: Callable,
                          inputs: Union[torch.Tensor, Sequence[torch.Tensor]],
                          modules: Optional[Union[nn.Module, Sequence[nn.Module]]] = (),
                          eps: float = 1e-6,
                          atol: float = 1e-5,
                          rtol: float = 1e-3,
                          grad_inputs=False,
                          gradgrad_inputs=False,
                          grad_params=False,
                          gradgrad_params=False):
    """Check grad and grad of grad wrt inputs and parameters of Modules.
    When `func` is vector-valued, the checks compare autodiff vjp against
    finite-difference vjp, where v is a sampled standard normal vector.
    This function is aimed to be as self-contained as possible so that it could
    be copied/pasted across different projects.
    Args:
        func (callable): A Python function that takes in a sequence of tensors
            (inputs) and a sequence of nn.Module (modules), and outputs a tensor
            or a sequence of tensors.
        inputs (sequence of Tensors): The input tensors.
        modules (sequence of nn.Module): The modules whose parameter gradient
            needs to be tested.
        eps (float, optional): Magnitude of two-sided finite difference
            perturbation.
        atol (float, optional): Absolute tolerance.
        rtol (float, optional): Relative tolerance.
        grad_inputs (bool, optional): Check gradients wrt inputs if True.
        gradgrad_inputs (bool, optional): Check gradients of gradients wrt
            inputs if True.
        grad_params (bool, optional): Check gradients wrt differentiable
            parameters of modules if True.
        gradgrad_params (bool, optional): Check gradients of gradients wrt
            differentiable parameters of modules if True.

    Returns:
        None.
    """

    if isinstance(inputs, torch.Tensor):
        inputs = (inputs,)

    if isinstance(modules, nn.Module):
        modules = (modules,)

    # Don't modify original objects.
    modules = tuple(copy.deepcopy(m) for m in modules)
    inputs = tuple(i.clone().requires_grad_() for i in inputs)

    func = _make_scalar_valued_func(func, inputs, modules)
    func_only_inputs = lambda *args: func(args, modules)  # noqa

    # Grad wrt inputs.
    if grad_inputs:
        torch.autograd.gradcheck(func_only_inputs, inputs, eps=eps, atol=atol, rtol=rtol)

    # Grad of grad wrt inputs.
    if gradgrad_inputs:
        torch.autograd.gradgradcheck(func_only_inputs, inputs, eps=eps, atol=atol, rtol=rtol)

    # Grad wrt params.
    if grad_params:
        params = [p for m in modules for p in m.parameters() if p.requires_grad]
        loss = func(inputs, modules)
        framework_grad = flatten(convert_none_to_zeros(torch.autograd.grad(loss, params, create_graph=True), params))

        numerical_grad = []
        for param in params:
            flat_param = param.reshape(-1)
            for i in range(len(flat_param)):
                flat_param[i] += eps  # In-place.
                plus_eps = func(inputs, modules).detach()
                flat_param[i] -= eps

                flat_param[i] -= eps
                minus_eps = func(inputs, modules).detach()
                flat_param[i] += eps

                numerical_grad.append((plus_eps - minus_eps) / (2 * eps))
                del plus_eps, minus_eps
        numerical_grad = torch.stack(numerical_grad)
        torch.testing.assert_allclose(numerical_grad, framework_grad, rtol=rtol, atol=atol)

    # Grad of grad wrt params.
    if gradgrad_params:
        def func_high_order(inputs_, modules_):
            params_ = [p for m in modules for p in m.parameters() if p.requires_grad]
            grads = torch.autograd.grad(func(inputs_, modules_), params_, create_graph=True, allow_unused=True)
            return tuple(grad for grad in grads if grad is not None)

        swiss_knife_gradcheck(func_high_order, inputs, modules, rtol=rtol, atol=atol, eps=eps, grad_params=True)


def _make_scalar_valued_func(func, inputs, modules):
    outputs = func(inputs, modules)
    output_size = outputs.numel() if torch.is_tensor(outputs) else sum(o.numel() for o in outputs)

    if output_size > 1:
        # Define this outside `func_scalar_valued` so that random tensors are generated only once.
        grad_outputs = tuple(torch.randn_like(o) for o in outputs)

        def func_scalar_valued(inputs_, modules_):
            outputs_ = func(inputs_, modules_)
            return sum((output * grad_output).sum() for output, grad_output, in zip(outputs_, grad_outputs))

        return func_scalar_valued

    return func


# EMA model averaging.
@torch.no_grad()
def ema_update(ema_model: nn.Module, model: nn.Module, gamma: Optional[float] = .999):
    if isinstance(model, nn.DataParallel):
        model = model.module  # Base model.

    ema_model_state = ema_model.training
    ema_model.eval()

    model_state = model.training
    model.eval()

    ema_state_dict = ema_model.state_dict()
    for key, val in model.state_dict().items():
        p1 = ema_state_dict[key]
        if val.dtype in (torch.int32, torch.int64):  # For `num_batches_tracked` in batch norm.
            p1.data.copy_(val.detach())
        else:
            p1.data.copy_(gamma * p1 + (1 - gamma) * val.detach())

    ema_model.train(ema_model_state)
    model.train(model_state)


class Module(nn.Module):
    def zero_grad(self) -> None:
        for p in self.parameters(): p.grad = None

    @property
    def device(self):
        return next(self.parameters()).device


# Meters.
class Meter(abc.ABC):
    @abc.abstractmethod
    def step(self, curr): raise NotImplementedError

    @property
    @abc.abstractmethod
    def val(self): raise NotImplementedError


class EMAMeter(Meter):
    """Standard exponential moving average."""

    def __init__(self, gamma: Optional[float] = .99):
        super(EMAMeter, self).__init__()
        self._val = None
        self._gamma = gamma
        self._history = []

    def step(self, x: Union[torch.Tensor, np.ndarray]):
        x = float(to_numpy(x))
        self._history.append(x)
        self._val = x if self._val is None else self._gamma * self._val + (1 - self._gamma) * x
        return self._val

    @property
    def val(self):
        return self._val

    @property
    def history(self):
        return self._history


class AverageMeter(Meter):
    """Exact online averaging."""

    def __init__(self):
        super(AverageMeter, self).__init__()
        self._val = 0.
        self.i = 0

    def step(self, x: Union[torch.Tensor, np.ndarray]):
        self._val = to_numpy(x) if self.i == 0 else self._val * self.i / (self.i + 1) + to_numpy(x) / (self.i + 1)
        self.i += 1
        return self._val

    @property
    def val(self):
        return self._val


def save_ckpt(model, ema_model, optimizer, path, scheduler=None, cloud_storage=False, epoch=None, global_step=None, best_acc=None, best_val=None, info=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state_dicts = {
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    if scheduler is not None:
        state_dicts["scheduler"] = scheduler.state_dict()
    if epoch is not None:
        state_dicts['epoch'] = epoch
    if best_acc is not None:
        state_dicts["best_acc"] = best_acc
    if best_val is not None:
        state_dicts["best_val_acc"] = best_val
    if info is not None:
        state_dicts["info"] = info
    if global_step is not None:
        state_dicts["global_step"] = global_step

    torch.save(state_dicts, path)

    logging.warning(f"Saved checkpoints at epoch {epoch} to {path}")
    
    if cloud_storage:
        gs_upload_from_path(path)


# Google cloud storage.
def gs_upload_from_path(local_path, remote_path=None, remove_local=True):
    # local_dir: Directory pointing to a local file, e.g. `/usr/local/google/home/lxuechen/bucket/tmp.txt`.
    # remote_dir: Directory pointing to a gcs path, e.g. `gs://lxuechen-bucket/tmp.txt`.
    if remote_path is None:
        remote_path = local_path
    _remote_dir = remote_path.replace('gs://', '')
    bucket_id = _remote_dir.split('/')[0]
    bucket_path = _remote_dir[len('{}/'.format(bucket_id)):]

    from google.cloud import storage  # noqa
    bucket = storage.Client().bucket(bucket_id)
    blob = bucket.blob(bucket_path)
    blob.upload_from_filename(local_path)  # The file is at `local_dir` locally.
    if remove_local:
        os.remove(local_path)


def gs_download_from_path(local_path, remote_path=None):
    if remote_path is None:
        remote_path = local_path
    _remote_dir = remote_path.replace('gs://', '')
    bucket_id = _remote_dir.split('/')[0]
    bucket_path = _remote_dir[len('{}/'.format(bucket_id)):]

    local_dir = os.path.dirname(local_path)
    if not os.path.exists(local_dir): os.makedirs(local_dir)
    from google.cloud import storage  # noqa
    bucket = storage.Client().bucket(bucket_id)
    blob = bucket.blob(bucket_path)
    blob.download_to_filename(local_path)  # The file is at `local_dir` locally.


def gs_download_from_dir(dir_):
    # dir: bucket-name/folder/subfolder.
    _remote_dir = dir_.replace('gs://', '')
    bucket_id = _remote_dir.split('/')[0]
    bucket_dir = _remote_dir[len('{}/'.format(bucket_id)):]

    from google.cloud import storage  # noqa
    bucket = storage.Client().bucket(bucket_id)
    blobs = bucket.list_blobs(prefix=bucket_dir)
    for blob in blobs:
        if blob.name.endswith('/'):  # Skip folders.
            continue
        # blob.name: folder/subfolder/file.
        tokens = blob.name.split('/')
        # Extract `local_dir` and `local_path`.
        local_dir_tokens = [bucket_id] + tokens[:-1]
        local_dir = os.path.join(*local_dir_tokens)

        local_path_tokens = [bucket_id] + tokens
        local_path = os.path.join(*local_path_tokens)
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        blob.download_to_filename(local_path)


def gs_file_exists(remote_path):
    _remote_dir = remote_path.replace('gs://', '')
    bucket_id = _remote_dir.split('/')[0]
    bucket_path = _remote_dir[len('{}/'.format(bucket_id)):]

    from google.cloud import storage  # noqa
    bucket = storage.Client().bucket(bucket_id)
    blob = bucket.blob(bucket_path)
    return blob.exists()


# Timer.
class Timer(object):
    def __init__(self, msg=None, stream: Optional[Union[str, io.IOBase]] = "stderr", logging=False, level=logging.WARN):
        super(Timer, self).__init__()
        self.msg = msg
        if isinstance(stream, str):
            stream = {
                "stderr": sys.stderr,
                "stdout": sys.stdout
            }[stream]
        else:
            if not isinstance(stream, io.IOBase):
                raise ValueError(f"Expected stream of type `io.IOBase`, but found: {type(stream)}")
        self.stream = stream  # Output stream.
        self.logging = logging
        self.level = level

    def __enter__(self):
        self.now = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        time_elapse = time.perf_counter() - self.now
        msg = f"Time elapse: {time_elapse:.6f}"
        if self.msg is not None:
            msg = f"{self.msg}: " + msg

        if self.logging:
            logging.log(level=self.level, msg=msg)
        else:
            print(msg, file=self.stream)


# Disable gc (e.g. for faster pickling).
# https://stackoverflow.com/questions/2766685/how-can-i-speed-up-unpickling-large-objects-if-i-have-plenty-of-ram
class DisableGC(object):
    def __init__(self):
        super(DisableGC, self).__init__()

    def __enter__(self):
        gc.disable()

    def __exit__(self, exc_type, exc_val, exc_tb):
        gc.enable()


# Checkpoint.
def all_ckpts(dir_, sort=True):
    # Returns all checkpoint paths in the form of a used-once generator.
    file_names = os.listdir(dir_)
    file_names = filter(lambda f: f.startswith('global_step_'), file_names)
    file_names = filter(lambda f: f.endswith('.ckpt'), file_names)
    file_names = map(lambda f: os.path.join(dir_, f), file_names)
    if sort: return sort_ckpts(file_names)
    return file_names


def sort_ckpts(file_names: Union[map, filter, list]):
    # Takes in an iterable (not necessarily a list); returns a list.
    if not isinstance(file_names, list):
        if not isinstance(file_names, collections.Iterable):
            raise ValueError
        file_names = list(file_names)
    # Avoid in-place ops that have side-effects.
    file_names_copy = file_names.copy()
    file_names_copy.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    return file_names_copy


def latest_ckpt(dir_):
    # Returns the path towards the latest ckpt. Returns `None` if no ckpt is found.
    # Assumes names are of the format `./parent_dir/global_step_i.ckpt`, where i is the index.
    # The prefix "global_step_" and suffix ".ckpt" must *both* be present in the path.
    def extract_id(name):
        assert isinstance(name, str)
        prefix, suffix = 'global_step_', '.ckpt'
        assert name.startswith('global_step_') and name.endswith('.ckpt')
        name = name[len(prefix):]
        name = name[:-len(suffix)]
        return int(name)

    file_names = os.listdir(dir_)
    file_names = filter(lambda f: f.startswith('global_step_'), file_names)
    file_names = filter(lambda f: f.endswith('.ckpt'), file_names)
    idx = map(extract_id, file_names)
    idx = list(idx)
    if len(idx) == 0:
        print(f'Did not find any checkpoints in: {dir_}')
        return None

    latest_path = os.path.join(dir_, f'global_step_{max(idx)}.ckpt')
    return latest_path


def select_activation(activation="softplus"):
    # Avoid materializing the objects; just return the constructors.
    return {
        "softplus": nn.Softplus,
        "swish": Swish,
        "mish": Mish,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "relu": lambda: nn.ReLU(inplace=True),
    }[activation]


def select_optimizer(optimizer):
    return {
        "adam": optim.Adam,
        "adadelta": optim.Adadelta,
        "sgd": optim.SGD,
        "adagrad": optim.Adagrad,
        "adamax": optim.Adamax
    }[optimizer]


# Data.
def get_data_stats(data_name):
    if data_name == "cifar10":
        input_size = (3, 32, 32)
        classes = 10
    elif data_name == "cifar100":
        input_size = (3, 32, 32)
        classes = 100
    elif data_name == "mnist":
        input_size = (1, 28, 28)
        classes = 10
    elif data_name == "svhn":
        input_size = (3, 32, 32)
        classes = 10
    elif data_name in ("imagenet32", "imagenet64", "celebahq", "celeba_5bit"):
        input_size = (3, 32, 32)
        classes = None
    else:
        raise ValueError(f"Unknown data: {data_name}")
    return {"input_size": input_size, "classes": classes}


def dequantize(x, nvals=256):
    """[0, 1] -> [0, nvals] -> add uniform noise -> [0, 1]"""
    noise = x.new().resize_as_(x).uniform_()
    x = x * (nvals - 1) + noise
    x = x / nvals
    return x

def compute_acc_bin(conf_thresh_lower, conf_thresh_upper, conf, pred, true):
    """
    Computes accuracy and average confidence for bin
    Args:
        conf_thresh_lower (float): Lower Threshold of confidence interval
        conf_thresh_upper (float): Upper Threshold of confidence interval
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
    Returns:
        (accuracy, avg_conf, len_bin): accuracy of bin, confidence of bin and number of elements in bin.
    """
    filtered_tuples = [x for x in zip(pred, true, conf) if x[2] > conf_thresh_lower and x[2] <= conf_thresh_upper]
    if len(filtered_tuples) < 1:
        return 0,0,0
    else:
        correct = len([x for x in filtered_tuples if x[0] == x[1]])  # How many correct labels
        len_bin = len(filtered_tuples)  # How many elements falls into given bin
        avg_conf = sum([x[2] for x in filtered_tuples]) / len_bin  # Avg confidence of BIN
        accuracy = float(correct)/len_bin  # accuracy of BIN
        return accuracy, avg_conf, len_bin

def ECE(conf, pred, true, bin_size=0.1):
    """
    Expected Calibration Error
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?
    Returns:
        ece: expected calibration error
    """
    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)  # Get bounds of bins

    n = len(conf)
    ece = 0  # Starting error

    for conf_thresh in upper_bounds:  # Go through bounds and find accuracies and confidences
        acc, avg_conf, len_bin = compute_acc_bin(conf_thresh-bin_size, conf_thresh, conf, pred, true)
        ece += np.abs(acc-avg_conf)*len_bin/n  # Add weigthed difference to ECE

    return ece

def MCE(conf, pred, true, bin_size = 0.1):
    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)

    cal_errors = []

    for conf_thresh in upper_bounds:
        acc, avg_conf, _ = compute_acc_bin(conf_thresh-bin_size, conf_thresh, conf, pred, true)
        cal_errors.append(np.abs(acc-avg_conf))

    return max(cal_errors)

def brier_score(targets, probs):
    if targets.ndim == 1:
        targets = targets[..., None]
        probs = probs[..., None]
    # for multi-class (not binary) classification
    return np.mean(np.sum((probs - targets)**2, axis=1))


def score_model(probs, y_true, verbose=False, normalize=False, bins=10):
    """
    Evaluate model using various scoring measures: Error Rate, ECE, MCE, NLL, Brier Score
    Params:
        probs: a list containing probabilities for all the classes with a shape of (samples, classes)
        y_true: a list containing the actual class labels
        verbose: (bool) are the scores printed out. (default = False)
        normalize: (bool) in case of 1-vs-K calibration, the probabilities need to be normalized.
        bins: (int) - into how many bins are probabilities divided (default = 15)
    Returns:
        (accuracy, error, ece, mce, brier, loss), returns scores dictionary
    """
    import sklearn.metrics as metrics
    from sklearn.metrics import log_loss, brier_score_loss

    preds = np.argmax(probs, axis=1)  # Take maximum confidence as prediction
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1) # Inputted targets are 1 hot

    if normalize:
        confs = np.max(probs, axis=1)/np.sum(probs, axis=1)
        # Check if everything below or equal to 1?
    else:
        confs = np.max(probs, axis=1)  # Take only maximum confidence

    accuracy = metrics.accuracy_score(y_true, preds) * 100
    error = 100 - accuracy

    # Calculate ECE
    ece = ECE(confs, preds, y_true, bin_size = 1/bins)
    # Calculate MCE
    mce = MCE(confs, preds, y_true, bin_size = 1/bins)
    
    # check for nans
    nan_mask = np.isnan(probs)
    probs = np.array(probs) # makes a copy, read only cannot be assigned
    probs[nan_mask] = -1. # 0.
    y_pred = probs
    # _loss = log_loss(y_true=y_true, y_pred=y_pred)
    # import pdb; pdb.set_trace()
    try:
      _loss = log_loss(y_true=y_true, y_pred=y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    except:
      _loss = 0.
    
    y_prob_true = np.array([y_pred[i, idx] for i, idx in enumerate(y_true)])  # Probability of positive class
    # y_prob_true = np.array([probs[i, idx] for i, idx in enumerate(y_true)])  # Probability of positive class
    brier = brier_score(y_true, y_prob_true)  # Brier Score (multiclass)

    if verbose:
        print("Accuracy:", accuracy)
        print("Error:", error)
        print("ECE:", ece)
        print("MCE:", mce)
        print("Loss:", _loss)
        print("brier:", brier)

    scores = {'acc': accuracy, 'error': error, 'ece': ece, 'mce': mce, 'brier': brier, 'loss': _loss}
    # logging.warning(f"scored model: {scores}")

    return (accuracy, error, ece, mce, brier, _loss)


def get_loader(data_name,
               root=None,
               train_batch_size=128,
               test_batch_size=1024,
               val_batch_size=1024,
               pin_memory=True,
               num_workers=8,
               train_transform=None,
               test_transform=None,
               drop_last=True,
               shuffle=True,
               data_aug=True,
               padding_mode="constant",
               subset=None,
               task="density"):
    if task not in ("density", "classification", "hybrid"):
        raise ValueError(f"Unknown task: {task}. Expected one of `density`, `classification`, `hybrid`.")
    logging.warning(f"Creating loaders for data {data_name} for task {task}")

    if root is None:
        root = os.path.join(os.path.expanduser("~"), 'datasets')

    if data_name in ('cifar10', 'cifar100'):
        if data_name == 'cifar10':
            mean, std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
        else:
            mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)

        if train_transform is None:
            if task in ("classification", "hybrid"):
                if data_aug:
                    train_transform = tv.transforms.Compose([
                        tv.transforms.RandomCrop(32, padding=4, padding_mode=padding_mode),
                        tv.transforms.RandomHorizontalFlip(),
                        tv.transforms.ToTensor(),
                        tv.transforms.Normalize(mean, std)
                    ])
                else:
                    train_transform = tv.transforms.Compose([
                        tv.transforms.ToTensor(),
                        tv.transforms.Normalize(mean, std)
                    ])
            else:  # `density`.
                if data_aug:
                    train_transform = tv.transforms.Compose([
                        tv.transforms.RandomCrop(32, padding=4, padding_mode=padding_mode),
                        tv.transforms.RandomHorizontalFlip(),
                        tv.transforms.ToTensor(),
                        dequantize,
                    ])
                else:
                    train_transform = tv.transforms.Compose([
                        tv.transforms.ToTensor(),
                        dequantize,
                    ])
        if test_transform is None:
            if task in ("classification", "hybrid"):
                test_transform = tv.transforms.Compose([
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize(mean, std)
                ])
            else:  # `density`.
                tv.transforms.Compose([
                    tv.transforms.ToTensor(),
                    dequantize
                ])

        if data_name == 'cifar10':
            train_data = tv.datasets.CIFAR10(root, transform=train_transform, train=True, download=True)
            test_data = tv.datasets.CIFAR10(root, transform=test_transform, train=False, download=True)
        else:
            train_data = tv.datasets.CIFAR100(root, transform=train_transform, train=True, download=True)
            test_data = tv.datasets.CIFAR100(root, transform=test_transform, train=False, download=True)

    elif data_name == "svhn":
        if train_transform is None:
            if task in ("classification", "hybrid"):
                if data_aug:
                    train_transform = tv.transforms.Compose([
                        tv.transforms.RandomCrop(32, padding=4, padding_mode=padding_mode),
                        tv.transforms.ToTensor(),
                    ])
                else:
                    train_transform = tv.transforms.Compose([tv.transforms.ToTensor()])
            else:  # `density`.
                if data_aug:
                    train_transform = tv.transforms.Compose([
                        tv.transforms.RandomCrop(32, padding=4, padding_mode=padding_mode),
                        tv.transforms.ToTensor(),
                        dequantize,
                    ])
                else:
                    train_transform = tv.transforms.Compose([
                        tv.transforms.ToTensor(),
                        dequantize
                    ])
        if test_transform is None:
            if task in ("classification", "hybrid"):
                test_transform = tv.transforms.Compose([
                    tv.transforms.ToTensor(),
                ])
            else:  # `density`.
                test_transform = tv.transforms.Compose([
                    tv.transforms.ToTensor(),
                    dequantize,
                ])
        train_data = tv.datasets.SVHN(root, transform=train_transform, split='train', download=True)
        test_data = tv.datasets.SVHN(root, transform=test_transform, split='test', download=True)

    elif data_name in ('mnist',):
        if train_transform is None:
            if task in ("classification", "hybrid"):
                train_transform = tv.transforms.Compose([tv.transforms.ToTensor()])
            else:  # `density`.
                train_transform = tv.transforms.Compose([
                    tv.transforms.ToTensor(),
                    dequantize
                ])
        if test_transform is None:
            if task in ("classification", "hybrid"):
                test_transform = tv.transforms.Compose([tv.transforms.ToTensor()])
            else:  # `density`.
                test_transform = tv.transforms.Compose([
                    tv.transforms.ToTensor(),
                    dequantize
                ])

        train_data = tv.datasets.MNIST(root, train=True, transform=train_transform, download=True)
        test_data = tv.datasets.MNIST(root, train=False, transform=test_transform, download=True)
    else:
        raise NotImplementedError(f"Unknown dataset: {data_name}.")

    logging.warning(f"train before split: {len(train_data)}")
    nval = int(len(train_data) * 0.1) # 10% of training set for validation
    train_data, val_data = torch.utils.data.random_split(train_data, [len(train_data) - nval, nval], generator=torch.Generator().manual_seed(42))
    train_loader = data.DataLoader(
        train_data,
        batch_size=train_batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers
    )
    if subset is not None:
        val_data = test_data
        # np permutate
        subset_idx = np.random.permutation(len(train_data))[:subset]
        sampler = data.sampler.SubsetRandomSampler(subset_idx)
        train_loader = data.DataLoader(
            train_data,
            batch_size=train_batch_size,
            drop_last=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
            sampler=sampler,
        )
        logging.warning(f'Created train subset of size {subset} for batch size {train_batch_size}: {len(train_loader)} batches')
    val_loader = data.DataLoader(
        val_data,
        batch_size=val_batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers
    )
    test_loader = data.DataLoader(
        test_data,
        batch_size=test_batch_size,
        drop_last=False,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers
    )
    logging.warning(f"train: {len(train_data)} | val: {nval} | test: {len(test_data)}")
    return train_loader, val_loader, test_loader

class Subset(object):

    def __init__(self, dataset, size):
        assert size <= len(dataset)
        self.dataset = dataset
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.dataset[index]


def count_examples(loader: data.DataLoader):
    """Count the number of examples in a dataloader."""
    count = 0
    for batch in loader:
        unpacked = batch
        while not torch.is_tensor(unpacked):
            unpacked = unpacked[0]
        count += unpacked.size(0)
    return count


class InfiniteLoader(object):
    """Wraps an existing loader so that it outputs stuff indefinitely; useful for semi-supervised learning."""

    def __init__(self, loader: data.DataLoader):
        super(InfiniteLoader, self).__init__()
        self.loader = loader
        self.iterator = iter(loader)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)
