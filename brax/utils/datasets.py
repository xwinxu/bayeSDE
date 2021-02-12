"""
Datasets for running our examples.
"""
from __future__ import absolute_import, division, print_function

import array
import gzip
import os
import struct
import urllib.request
from os import path

import jax
import jax.numpy as np
import jax.random as random
import numpy as onp
import numpy.random as npr
import tensorflow as tf
import tensorflow_datasets as tfds
from brax.utils.registry import add_data
from jax.api import vmap
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST

tf.config.experimental.set_visible_devices([], "GPU")

from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).parent.absolute()
def get_dataset(batch_size, test_batch_size, dataset):
    if dataset == "mnist":
        input_size = (28, 28, 1)
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        train_set = MNIST(f"{SCRIPT_DIR}/data/mnist", train=True, transform=transform, download=True)
        test_set = MNIST(f"{SCRIPT_DIR}/data/mnist", train=False, transform=transform, download=True)
    elif dataset == "cifar10":
        input_size = (32, 32, 3)
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]
        )
        train_set = CIFAR10(
            f"{SCRIPT_DIR}/data/cifar10",
            train=True,
            transform=transform,
            download=True,
        )
        test_set = CIFAR10(
            f"{SCRIPT_DIR}/data/cifar10",
            train=False,
            transform=test_transform,
            download=True,
        )


    nval = int(len(train_set) * 0.1)
    train_set, val_set = torch.utils.data.random_split(train_set, [len(train_set) - nval, nval], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    train_eval_loader = DataLoader(train_set, batch_size=test_batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=test_batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False)
    return train_loader, train_eval_loader, val_loader, test_loader, input_size, len(train_set)

@add_data('cos')
def build_toy_dataset(key, test_n_data, batch_size, x_lim=[0, 6],  n_data=40, noise_std=0.1):
  """Time series data with gap for visualizing uncertainty.
  """
  rs = npr.RandomState(0)
  inputs  = np.concatenate([np.linspace(x_lim[0], 2, num=n_data//2),
                            np.linspace(x_lim[-1], 8, num=n_data//2)])
  true_fn = lambda x: np.cos(x)
  targets = true_fn(inputs) + rs.randn(n_data) * noise_std
  inputs = (inputs - 4.0) / 4.0
  inputs  = inputs.reshape((len(inputs), 1)) # (40, 1)
  targets = targets.reshape((len(targets), 1)) # (40, 1)

  # test set
  D = inputs.shape[-1]
  test_x0 = np.repeat(
      onp.expand_dims(np.linspace(x_lim[0] - 2, x_lim[1] + 2, test_n_data), axis=1), D, axis=1)  # (N, D)
  test_x0 = (test_x0 - 4.0) / 4.0
  test_x1 = onp.repeat(np.expand_dims(true_fn(test_x0[:, 0]), axis=1), D, axis=1) # (N, D)
  test = (test_x0, test_x1)

  def get_batch(key, test_n_data, batch_size, D=1):
    assert test_n_data % batch_size == 0
    num_batches = test_n_data / batch_size
    batch_x = onp.random.uniform(size=(batch_size, D), low=x_lim[0], high=x_lim[1]) # (bs, D)
    batch_x = (batch_x - 4.0) / 4.0
    batch_y = onp.repeat(true_fn(batch_x[:, 0])[..., None], D, axis=1) # (bs, D)

    return 1, (batch_x, batch_y)

  return inputs, targets, test, get_batch, noise_std

@add_data('coscos')
def coscos(key, test_n_data, batch_size, x_lim=[0, 5],  n_data=10, noise_std=0.1):
  """Time series data with gap for visualizing uncertainty.
  """
  rs = npr.RandomState(0)
  inputs1  = np.concatenate([np.linspace(x_lim[0], 2, num=n_data//2),
                            np.linspace(x_lim[-1] - 3, 4, num=n_data//2)])
  inputs2  = np.concatenate([np.linspace(x_lim[0] + 1, 2, num=n_data//2),
                            np.linspace(x_lim[-1] - 3, 4, num=n_data//2)])
  true_fn1 = lambda x: np.sqrt(x+1)
  true_fn2 = lambda x: -np.sqrt(x+1)
  targets1 = true_fn1(inputs1) + rs.randn(n_data) * noise_std
  targets1 = targets1.reshape((len(targets1), 1)) # (40, 1)
  targets2 = true_fn2(inputs2) + rs.randn(n_data) * noise_std
  targets2 = targets2.reshape((len(targets2), 1))
  targets = np.concatenate([targets1, targets2])
  inputs1 = inputs1.reshape((len(inputs1), 1)) # (40, 1)
  inputs2 = inputs2.reshape((len(inputs2), 1)) # (40, 1)
  inputs = np.concatenate([inputs1, inputs2])

  # test set
  D = inputs1.shape[-1]
  test_x0 = np.repeat(
      onp.expand_dims(np.linspace(x_lim[0] - 2, x_lim[1] + 2, test_n_data), axis=1), D, axis=1)  # (N, D)
  test_x11 = onp.repeat(np.expand_dims(true_fn1(test_x0[:, 0]), axis=1), D, axis=1) # (N, D)
  test_x12 = onp.repeat(np.expand_dims(true_fn2(test_x0[:, 0]), axis=1), D, axis=1) # (N, D)
  test = (np.concatenate([test_x0, test_x0]), np.concatenate([test_x11, test_x12]))

  def get_batch(key, test_n_data, batch_size, D=1):
    assert test_n_data % batch_size == 0
    num_batches = test_n_data / batch_size
    batch_x = onp.random.uniform(size=(batch_size, D), low=x_lim[0], high=x_lim[1]) # (bs, D)
    batch_y = onp.repeat(true_fn1(batch_x[:, 0])[..., None], D, axis=1) # (bs, D)

    return 1, (batch_x, batch_y)

  return inputs, targets, test, get_batch, noise_std

@add_data('cos2')
def cos2(key, test_n_data, batch_size, x_lim=[0, 6],  n_data=40, noise_std=0.1):
  """Neg cos overlayed.
  """
  rs = npr.RandomState(0)
  inputs  = np.concatenate([np.linspace(x_lim[0], 2, num=n_data//2),
                            np.linspace(x_lim[-1], 8, num=n_data//2)])
  true_fn1 = lambda x: np.cos(x)
  true_fn2 = lambda x: -np.cos(x)
  targets1 = true_fn1(inputs) + rs.randn(n_data) * noise_std
  targets1 = targets1.reshape((len(targets1), 1)) # (40, 1)
  targets2 = true_fn2(inputs) + rs.randn(n_data) * noise_std
  targets2 = targets2.reshape((len(targets2), 1))
  targets = np.concatenate([targets1, targets2])
  inputs = (inputs - 4.0) / 4.0
  inputs = inputs.reshape((len(inputs), 1)) # (40, 1)
  inputs = np.concatenate([inputs, inputs])

  # test set
  D = inputs.shape[-1]
  test_x0 = np.repeat(
      onp.expand_dims(np.linspace(x_lim[0] - 2, x_lim[1] + 2, test_n_data), axis=1), D, axis=1)  # (N, D)
  test_x0 = (test_x0 - 4.0) / 4.0
  test_x11 = onp.repeat(np.expand_dims(true_fn1(test_x0[:, 0]), axis=1), D, axis=1) # (N, D)
  test_x12 = onp.repeat(np.expand_dims(true_fn2(test_x0[:, 0]), axis=1), D, axis=1) # (N, D)
  test = (np.concatenate([test_x0, test_x0]), np.concatenate([test_x11, test_x12]))

  def get_batch(key, test_n_data, batch_size, D=1):
    assert test_n_data % batch_size == 0
    num_batches = test_n_data / batch_size
    batch_x = onp.random.uniform(size=(batch_size, D), low=x_lim[0], high=x_lim[1]) # (bs, D)
    batch_x = (batch_x - 4.0) / 4.0
    batch_y = onp.repeat(true_fn1(batch_x[:, 0])[..., None], D, axis=1) # (bs, D)

    return 1, (batch_x, batch_y)

  return inputs, targets, test, get_batch, noise_std

_DATA = "/tmp/jax_example_data/"

def _download(url, filename):
  """Download a url to a file in the JAX data temp directory."""
  if not path.exists(_DATA):
    os.makedirs(_DATA)
  out_file = path.join(_DATA, filename)
  if not path.isfile(out_file):
    urllib.request.urlretrieve(url, out_file)
    print("downloaded {} to {}".format(url, _DATA))


def _partial_flatten(x):
  """Flatten all but the first dimension of an ndarray."""
  return np.reshape(x, (x.shape[0], -1))


def _one_hot(x, k, dtype=np.float32):
  """Create a one-hot encoding of x of size k."""
  return np.array(x[:, None] == np.arange(k), dtype)


def mnist_raw():
  """Download and parse the raw MNIST dataset."""
  # CVDF mirror of http://yann.lecun.com/exdb/mnist/
  base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

  def parse_labels(filename):
    with gzip.open(filename, "rb") as fh:
      _ = struct.unpack(">II", fh.read(8))
      return np.array(array.array("B", fh.read()), dtype=np.uint8)

  def parse_images(filename):
    with gzip.open(filename, "rb") as fh:
      _, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
      return np.array(array.array("B", fh.read()),
                      dtype=np.uint8).reshape(num_data, rows, cols)

  for filename in ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                   "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]:
    _download(base_url + filename, filename)

  train_images = parse_images(path.join(_DATA, "train-images-idx3-ubyte.gz"))
  train_labels = parse_labels(path.join(_DATA, "train-labels-idx1-ubyte.gz"))
  test_images = parse_images(path.join(_DATA, "t10k-images-idx3-ubyte.gz"))
  test_labels = parse_labels(path.join(_DATA, "t10k-labels-idx1-ubyte.gz"))

  return train_images, train_labels, test_images, test_labels


def mnist(permute_train=False):
  """Download, parse and process MNIST data to unit scale and one-hot labels."""
  train_images, train_labels, test_images, test_labels = mnist_raw()

  train_images = _partial_flatten(train_images) / np.float32(255.)
  test_images = _partial_flatten(test_images) / np.float32(255.)
  train_labels = _one_hot(train_labels, 10)
  test_labels = _one_hot(test_labels, 10)

  if permute_train:
    perm = onp.random.RandomState(0).permutation(train_images.shape[0])
    train_images = train_images[perm]
    train_labels = train_labels[perm]

  return train_images, train_labels, test_images, test_labels

@add_data('mnist_flat')
def mnist_dataset(seed, batch_size, permute_train=False):
  """MNIST dataset processed to unit scale and one-hot labels"""
  train_images, train_labels, test_images, test_labels = mnist(permute_train=permute_train)
  import pdb
  pdb.set_trace()
  num_train = train_images.shape[0]
  num_batches, leftover = divmod(num_train, batch_size)
  num_batches += bool(leftover)
  num_test_batches, leftover = divmod(test_images.shape[0], batch_size)
  num_test_batches += bool(leftover)

  def get_batch(seed):
    rng = npr.RandomState(seed)
    while True:
      perm = rng.permutation(num_train)
      for i in range(num_batches):
        batch_idx = perm[i * batch_size: (i + 1) * batch_size]
        yield (train_images[batch_idx], train_labels[batch_idx]), (test_images[batch_idx], test_labels[batch_idx])

  batches = get_batch(seed)
  meta = {
    'num_batches': num_batches,
    'num_test_batches': num_test_batches
  }
  # (60000, 784) (60000, 10) (10000, 784) (10000, 10)
  return train_images, train_labels, test_images, test_labels, batches, meta

@add_data('mnist')
def load_mnist(seed, batch_size, test_batch_size, num_epochs):
  """
  Initialize data.
  """
  (ds_train, ds_test), ds_info = tfds.load('mnist',
                                            split=['train', 'test'],
                                            shuffle_files=True,
                                            with_info=True,
                                            as_supervised=True,
                                            read_config=tfds.ReadConfig(shuffle_seed=seed,
                                                                        try_autocache=False))

  num_train = ds_info.splits['train'].num_examples
  num_test = ds_info.splits['test'].num_examples

  # make sure all batches are the same size to minimize jit compilation cache
  assert num_train % batch_size == 0
  num_batches, _ = divmod(num_train, batch_size)
  assert num_test % test_batch_size == 0
  num_test_batches, _ = divmod(num_test, test_batch_size)

  # ds_train = ds_train.cache().repeat().shuffle(1000, seed=seed).batch(batch_size)
  ds_train = ds_train.shuffle(1000, seed=seed).repeat().batch(batch_size)
  ds_test_eval = ds_test.batch(test_batch_size).repeat()

  ds_train, ds_test_eval = tfds.as_numpy(ds_train), tfds.as_numpy(ds_test_eval)
  # image, label = tfds.as_numpy(ds_train.take(1))

  meta = {
      "num_batches": num_batches,
      "num_test_batches": num_test_batches,
      "num_training_samples": num_train
  }

  return ds_train, ds_test_eval, meta

################## 1D regression #######################

def sample_noise(key, n_data):
  """Gaussian noise."""
  rngs = jax.random.split(key, n_data)
  noise = vmap(random.normal, (0, None), 0)(rngs, (1,))
  assert len(noise) == n_data
  return noise.squeeze()

@add_data('b20')
def b20_toy1d_dataset(key, test_n_data, batch_size, n_data=20, x_lim=[-2, 2], noise_std=3):
  subkeys = jax.random.split(key, n_data)
  inputs = np.concatenate([np.linspace(x_lim[0], -0.5, num=n_data//2),
                          np.linspace(0, x_lim[-1], num=n_data//2)])
  noise = sample_noise(key, n_data) * noise_std
  true_fn = lambda x: x**3
  targets = true_fn(inputs) + noise
  inputs, targets = inputs[..., None], targets[..., None]
  print('plot sample-20 inputs: {} targets: {}'.format(inputs.shape, targets.shape))

  # test set for evaluation
  D = inputs.shape[-1]
  print("D", D)
  test_x0 = np.repeat(
      np.expand_dims(np.linspace(x_lim[0] - 2, x_lim[1] + 2, test_n_data), axis=1), D, axis=1)  # (N, D)
  test_x1 = np.repeat(np.expand_dims(true_fn(test_x0[:, 0]), axis=1), D, axis=1) # (N, D)
  test = (test_x0, test_x1)

  def get_batch(key, test_n_data, batch_size, D=1):
    assert test_n_data % batch_size == 0
    num_batches = test_n_data / batch_size
    # return key so that we don't affect training potentially
    key, subkey = jax.random.split(key)
    batch_x = jax.random.uniform(subkey, (batch_size, D), minval=x_lim[0], maxval=x_lim[1]) # (bs, D)
    batch_y = np.repeat(true_fn(batch_x[:, 0])[..., None], D, axis=1) # (bs, D)

    return key, (batch_x, batch_y)

  return inputs, targets, test, get_batch, noise_std

@add_data('b40')
def b40_toy1d_dataset(key, data_size, batch_size, n_data=40, x_lim=[-4, 4], noise_std=3):
  """x^3 invertible function with added Gaussian noise.
  Return:
    (inputs, targets) are for plotting/visualization
    get_batch: function for getting another randomly selected batch of data
  """
  subkeys = jax.random.split(key, n_data)
  inputs = np.concatenate([np.linspace(-4, -3, num=n_data//2),
                          np.linspace(-2.5, -1, num=n_data//5),
                          np.linspace(-1, 2, num=n_data//5),
                          np.linspace(2, 4, num=n_data//2)])
  noise = sample_noise(key, inputs.shape[0])
  # print(noise)
  true_fn = lambda x: x ** 3
  targets = true_fn(inputs) + noise * noise_std
  inputs = inputs[..., None]
  targets = targets[..., None]
  print('plot sample-gap-40 inputs: {} targets: {}'.format(inputs.shape, targets.shape))

  # test set for evaluation
  D = inputs.shape[-1]
  test_x0 = np.repeat(
      np.expand_dims(np.linspace(x_lim[0] - 2, x_lim[1] + 2, data_size), axis=1), D, axis=1)  # (N, D)
  test_x1 = np.repeat(np.expand_dims(true_fn(test_x0[:, 0]), axis=1), D, axis=1) # (N, D)
  test = (test_x0, test_x1)

  def get_batch(key, data_size, batch_size, D=1):
    assert data_size % batch_size == 0
    num_batches = data_size / batch_size
    # return key so that we don't affect training potentially
    key, subkey = jax.random.split(key)
    batch_x = jax.random.uniform(subkey, (batch_size, D), minval=x_lim[0], maxval=x_lim[1]) # (bs, D)
    batch_y = np.repeat(true_fn(batch_x[:, 0])[..., None], D, axis=1) # (bs, D)

    return key, (batch_x, batch_y)

  return inputs, targets, test, get_batch, noise_std

@add_data('b40gap')
def b40_toy1d_dataset(key, data_size, batch_size, x_lim=[-4, 4], n_data=40, noise_std=3):
  """x^3 invertible function with added Gaussian noise.
  Return:
    (inputs, targets) are for plotting/visualization
    get_batch: function for getting another randomly selected batch of data
    noise_std: 3 based on a paper's recommendations
  """
  subkeys = jax.random.split(key, n_data)
  inputs = np.concatenate([np.linspace(-4, -3, num=n_data//4),
                          np.linspace(-2.5, -1, num=n_data//4),
                          np.linspace(1, 2, num=n_data//4),
                          np.linspace(3, 4, num=n_data//4)])
  noise = sample_noise(key, n_data) # n_data if split uniformly
  # print(noise)
  true_fn = lambda x: x ** 3
  targets = true_fn(inputs) + noise * noise_std
  inputs = inputs[..., None]
  targets = targets[..., None]
  print('plot sample-40 inputs: {} targets: {}'.format(inputs.shape, targets.shape))

  # test set for evaluation
  D = inputs.shape[-1]
  test_x0 = np.repeat(
      np.expand_dims(np.linspace(x_lim[0], x_lim[1], data_size), axis=1), D, axis=1)  # (N, D)
  test_x1 = np.repeat(np.expand_dims(true_fn(test_x0[:, 0]), axis=1), D, axis=1) # (N, D)
  test = (test_x0, test_x1)

  def get_batch(key, data_size, batch_size, D=1):
    assert data_size % batch_size == 0
    num_batches = data_size / batch_size
    key, subkey = jax.random.split(key)
    batch_x = jax.random.uniform(subkey, (batch_size, D), minval=x_lim[0], maxval=x_lim[1]) # (bs, D)
    batch_y = np.repeat(true_fn(batch_x[:, 0])[..., None], D, axis=1) # (bs, D)

    return key, (batch_x, batch_y)

  return inputs, targets, test, get_batch, noise_std

if __name__ == "__main__":
  key = random.PRNGKey(0)
  train_images, train_labels, test_images, test_labels, batches = mnist_dataset(key, 128)
  print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape, next(batches))
