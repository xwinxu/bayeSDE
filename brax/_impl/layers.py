import functools
import operator as op
from functools import partial

import jax
import jax.lax as lax
import jax.numpy as np
import jax.random as random
from jax.experimental import stax
from jax.experimental.stax import elementwise
from jax.flatten_util import ravel_pytree
from jax.nn.initializers import glorot_normal, he_normal, normal, ones, zeros
from jax.tree_util import tree_map

from brax.utils.registry import *
from brax.utils.sdeint import _sdeint, sdeint
from brax.utils.utils import he_normal_stdev


def augmented_post_drift(rng, state_unravel_fn, param_unravel_fn, driftx_apply_fn, prior_driftw_apply_fn, driftw0_apply_fn,
                          driftw_apply_fn, diff_apply_fn, W_driftwt, W_priorwt, W_diffusion, W0_post, setting, t, aug, eps, **kwargs):
  """augmented state: [x, w, kl]
  """
  flat_w, x, kl = state_unravel_fn(aug)  
  w = param_unravel_fn(flat_w) 

  dxt, _ = driftx_apply_fn(w, (x, t), **kwargs) 

  dwtkl, _ = driftw_apply_fn(W_driftwt, (flat_w, t), **kwargs)
  
  pw_drift, _ = prior_driftw_apply_fn(W_priorwt, (flat_w, t), **kwargs) 
  diffwt, _ = diff_apply_fn(W_diffusion, (flat_w, t), **kwargs) 


  # stop gradient only to compute the KL
  _dwtkl = dwtkl 
  u = (_dwtkl - pw_drift) * 0
  if setting['diffw']:
    u = (_dwtkl - pw_drift) / diffwt
  dkl = np.sum((u ** 2) / 2) 
  u2 = u * eps
  dkl_stl = np.sum(u2)

  if setting['stl']:
    if setting['priorw0_sigma'] >= 0:
      W0_post = tree_map(lambda w: lax.stop_gradient(w), W0_post)
    if setting['diff_drift']:
      W_driftwtkl = tree_map(lambda w: lax.stop_gradient(w), W_driftwt[-1])
      W_driftwtkl = (W_driftwt[0], W_driftwtkl)
      _dwtkl, _ = driftw_apply_fn(W_driftwtkl, (flat_w, t), **kwargs) 
    else:
      W_driftwtkl = tree_map(lambda w: lax.stop_gradient(w), W_driftwt)
      _dwtkl, _ = driftw_apply_fn(W_driftwtkl, (flat_w, t), **kwargs) 

  if setting['stl']:
    u2 = (_dwtkl - pw_drift) * 0
    if setting['diffw']:
      u2 = (_dwtkl - pw_drift) / diffwt
    u2 = u2 * eps # u dBt
    dkl_stl = np.sum(u2)
  dkl = dkl + dkl_stl

  outputs = [dwtkl, dxt, dkl]

  return ravel_pytree(outputs)[0]

def augmented_prior_drift(state_unravel_fn, param_unravel_fn, driftx_apply_fn, prior_driftw_apply_fn, diff_apply_fn, 
                          W_priorwt, W_diffusion, setting, t, aug, eps, **kwargs): # partial last 3
  flat_w, x, kl = state_unravel_fn(aug)  # flat_w inputs
  w = param_unravel_fn(flat_w)  # hierarchical w params

  dxt, _ = driftx_apply_fn(w, (x, t), **kwargs)
  dwtkl, _ = prior_driftw_apply_fn(W_priorwt, (flat_w, t), **kwargs)

  pw_drift = 0.
  diffwt, _ = diff_apply_fn(W_diffusion, (flat_w, t), **kwargs)  # with time dependence

  u = (dwtkl - pw_drift) * 0
  if setting['diffw']:
    u = (dwtkl - pw_drift) / diffwt
  dkl = np.sum((u ** 2) / 2)

  # new FMC formulation
  u2 = u * eps
  dkl_fmc = np.sum(u2)
  dkl = dkl + dkl_fmc

  outputs = [dwtkl, dxt, dkl]
  return ravel_pytree(outputs)[0]

def augmented_diffusion(state_unravel_fn, param_unravel_fn, diff_apply_fn, W_diffusion, t, aug, **kwargs): #partial last 3
  flat_w, x, kl = state_unravel_fn(aug)
  w = param_unravel_fn(flat_w)

  gxt = np.zeros_like(x)
  gwt, _ = diff_apply_fn(W_diffusion, (flat_w, t), **kwargs)
  gkl = np.zeros_like(kl)

  outputs = [gwt, gxt, gkl]
  return ravel_pytree(outputs)[0]

def aug_init(W0, _inputs): # partial over last
  flat_aug, _ = ravel_pytree([W0, _inputs, 0.])
  flat_w_dim = len(ravel_pytree([W0])[0])
  return flat_aug, flat_w_dim

def SDELayer(driftx_layer, prior_driftw_layer, post_driftw_layer, diffusion_layer, x_dim, w_dim, setting):

  # add posterior for w0 potentially
  driftw0_layer, driftw0_init_fn, driftw0_apply_fn = None, None, None
  if setting['priorw0_sigma'] >= 0:
    print("Being Bayesian about W0...")
    driftw0_layer = BayesianLayer(driftx_layer)
    driftw0_init_fn, driftw0_apply_fn = driftw0_layer

  driftx_init_fn, driftx_apply_fn = driftx_layer
  prior_driftw_init_fn, prior_driftw_apply_fn = prior_driftw_layer  # Assume prior has no params
  driftw_init_fn, driftw_apply_fn = post_driftw_layer
  diff_init_fn, diff_apply_fn = diffusion_layer

  def init_fun(rng, input_shape):
    output_shape = input_shape
    k1, k2, k3, k4 = random.split(rng, 4)
    W0, _ = driftx_init_fn(k1, input_shape)
    W_driftwt, _ = driftw_init_fn(k2, (-1, w_dim))
    W_difft, _ = diff_init_fn(k3, (-1, w_dim))
    W_priorwt, _ = prior_driftw_init_fn(k4, (-1, w_dim))

    W_driftwt = tree_map(lambda w: w * setting['driftw_scale'], W_driftwt)

    if driftw0_layer:
      k5, k6 = random.split(rng)
      W0_post, _ = driftw0_init_fn(k5, input_shape)
      if setting['priorw0_sigma'] >= 0:
        # set the prior distribution to be same as he initialization stdev
        setting['priorw0_sigma'] = he_normal_stdev(k6, (input_shape[-1], 1)) 
    else:
      W0_post = None

    return output_shape, (W0, W_driftwt, W_difft, W_priorwt, W0_post)

  def apply_fun(params, inputs, rng, **kwargs):
    """(aka predict_apply_fn)
    Args:
      inputs: x_t0 ... x_tn
      params: (w0, w_driftwt, w_difft)
    Return: 
      x_t1, log p(x) / q(x)
    """
    W0, W_driftwt, W_diffusion, W_priorwt, W0_post = params

    # Bayesian about W0 (mean field)
    W0kl = 0.
    if W0_post is not None:
      tw0 = 0 
      _, W0, W0kl = driftw0_apply_fn(W0_post, (inputs, tw0), rng=rng, logsigma2=np.log(setting['priorw0_sigma']))
    flat_W0, param_unravel_fn = ravel_pytree(W0)
    _, state_unravel_fn = ravel_pytree([flat_W0, inputs, W0kl])

    aug_init_ = lambda inputs_: aug_init(W0, inputs_)
    aug_post_drift = lambda t_, aug_, eps_: augmented_post_drift(rng, state_unravel_fn, param_unravel_fn, driftx_apply_fn, prior_driftw_apply_fn, driftw0_apply_fn,
                                    driftw_apply_fn, diff_apply_fn, W_driftwt, W_priorwt, W_diffusion, W0_post, setting, t_, aug_, eps_, **kwargs)
    aug_prior_drift =lambda t_, aug_, eps_: augmented_prior_drift(state_unravel_fn, param_unravel_fn, driftx_apply_fn, prior_driftw_apply_fn,
                                    diff_apply_fn, W_priorwt, W_diffusion, setting, t_, aug_, eps_, **kwargs)
    aug_diffusion = lambda t_, aug_: augmented_diffusion(state_unravel_fn, param_unravel_fn, diff_apply_fn, W_diffusion, t_, aug_, **kwargs)

    if kwargs.get('entire', False):
      print("returning entire trajectory...")
      outputs = sdeint(aug_post_drift, aug_diffusion, aug_init_(inputs)[0], aug_init_(inputs)[1], np.linspace(0, 1, 90), rng) 
      ws, xs, kls = zip(*[state_unravel_fn(out) for out in outputs]) # list(x_dim), list(w_dim), list(1)
      prior_outputs = sdeint(aug_prior_drift, aug_diffusion, aug_init_(inputs)[0], aug_init_(inputs)[1], np.linspace(0, 1, 90), rng) 
      prior_ws, prior_xs, prior_kls = zip(*[state_unravel_fn(out) for out in prior_outputs]) # list(x_dim), list(w_dim), list(1)
      return xs, ws, kls, prior_xs, prior_ws, prior_kls

    solution = _sdeint(aug_post_drift, aug_diffusion, aug_init_(inputs)[0], aug_init_(inputs)[1], np.linspace(0, 1, 20), rng)
    w, x, kl = state_unravel_fn(solution)
    prior_outputs = _sdeint(aug_prior_drift, aug_diffusion, aug_init_(inputs)[0], aug_init_(inputs)[1], np.linspace(0, 1, 20), rng) 
    prior_ws, prior_xs, prior_kls = state_unravel_fn(prior_outputs)
    return x, w, kl, prior_xs, prior_ws, prior_kls

  return init_fun, apply_fun


def Const(const=1e-3):
  def init_fun(rng, input_shape):
    return input_shape, ()

  def apply_fun(params, inputs, **kwargs):
    return np.ones_like(inputs) * const

  return init_fun, apply_fun


def Affine(mult=0., const=1e-3):
  def init_fun(rng, input_shape):
    # no trainable parameters
    return input_shape, ()

  def apply_fun(params, inputs, **kwargs):
    return inputs * mult + const

  return init_fun, apply_fun


def Additive(layer1, layer2):
  """
  Learns the difference of two layers.
  layer1: prior drift
  layer2: posterior drift layer
  """
  init_fn1, apply_fn1 = layer1
  init_fn2, apply_fn2 = layer2

  def init_fun(rng, input_shape):
    output1, params1 = init_fn1(rng, input_shape)
    output2, params2 = init_fn2(rng, input_shape)
    return output1, (params1, params2)

  def apply_fun(params, inputs, **kwargs):
    params1, params2 = params
    t1, out1 = apply_fn1(params1, inputs, **kwargs)
    t2, out2 = apply_fn2(params2, inputs, **kwargs)
    return t2, out2 + out1

  return init_fun, apply_fun


# activations
rbf = lambda x: np.exp(-x ** 2)
Rbf = elementwise(rbf)
Rbf = register('rbf')(Rbf) # @register('rbf')
Elu = elementwise(jax.nn.elu)
Elu = register('elu')(stax.Elu)
Softplus = register('softplus')(stax.Softplus)
swish = lambda x: x * jax.nn.sigmoid(x)
Swish = elementwise(swish)
Swish = register('swish_nobeta')(Swish)
Relu = register('relu')(stax.Relu)
Tanh = register('tanh')(stax.Tanh)

@register('swish')
def Swish_(out_dim, beta_init=0.5):
  """
  Trainable Swish function to learn 
  """
  swish = lambda beta, x: np.array(x) * jax.nn.sigmoid(np.array(x) * jax.nn.softplus(beta)) # no / 1.1 for lipschitz

  def init_fun(rng, input_shape):
    output_shape = input_shape[:-1] + (out_dim,)
    beta0 = np.ones((out_dim,)) * beta_init
    return input_shape, (np.array(beta0),)

  def apply_fun(params, inputs, **kwargs):
    beta_, = params
    ret = swish(beta_, inputs)
    return ret
  
  return init_fun, apply_fun


def Flatten():
  """
  Flattens all but leading dim in input of convolutional layer e.g. (100, 28, 28, 1) -> (100, 784)
  """
  def init_fun(rng, input_shape):
    output_shape = (input_shape[0], functools.reduce(op.mul, input_shape[1:], 1))
    return output_shape, ()

  def apply_fun(params, inputs, **kwargs):
    x = inputs
    x_flat = x.reshape((x.shape[0], -1))
    return x_flat

  return init_fun, apply_fun


def _augment(x, pad):
  """Augment inputs by padding zeros to final dimension.
  e.g. input dimension in 1D, number of channels in 2D conv
  """
  z = np.zeros(x.shape[:-1] + (1 * pad,))
  return np.concatenate([x, z], axis=-1)

def AugmentedLayer(pad_zeros):
  """
  Wrapper for augmented neural ODE.
  """
  def init_fun(rng, input_shape):
    output_shape = input_shape[:-1] + (pad_zeros + input_shape[-1],)
    return output_shape, ()

  def apply_fun(params, inputs, **kwargs):
    x = inputs
    xzeros = _augment(x, pad_zeros)
    return xzeros
  
  return init_fun, apply_fun

def shape_dependent(make_layer):
  """
  Args:
    make_layer: a one-argument function that takes an input shape as an argument
      (a tuple of positive integers) and returns an (init_fun, apply_fun) pair.
  Returns:
    A new layer, meaning an (init_fun, apply_fun) pair, representing the same
    layer as returned by `make_layer` but with its construction delayed until
    input shapes are known.
  """
  def init_fun(rng, input_shape):
    return make_layer(input_shape)[0](rng, input_shape)
  def apply_fun(params, inputs, **kwargs):
    t, x = inputs
    return make_layer(x.shape)[1](params, inputs, **kwargs)

  return init_fun, apply_fun

def const(shape, c):
  return np.ones(shape) * c

def normal_kldiv(mu1, mu2, logsigma1, logsigma2):
  return (logsigma2 - logsigma1) \
          + (np.exp(logsigma1)**2 + (mu1 - mu2)**2) * np.exp(-2 * logsigma2) / 2 - 0.5

def Dense(out_dim, W_init=he_normal(), b_init=normal(), rho_init=partial(const, c=-5)):
  """Layer constructor function for a dense (fully-connected) Bayesian linear layer."""
  def init_fun(rng, input_shape):
    output_shape = input_shape[:-1] + (out_dim,)
    k1, k2, k3, k4 = random.split(rng, 4)
    W_mu, b_mu = W_init(k1, (input_shape[-1], out_dim)), b_init(k2, (out_dim,))
    W_rho, b_rho = rho_init((input_shape[-1], out_dim)), rho_init((out_dim,))
    return output_shape, (W_mu, b_mu, W_rho, b_rho)

  def apply_fun(params, inputs, rng, **kwargs):
    inputs, kl = inputs
    subkeys = random.split(rng, 2)

    W_mu, b_mu, W_rho, b_rho = params
    W_eps = random.normal(subkeys[0], W_mu.shape)
    b_eps = random.normal(subkeys[1], b_mu.shape)
    # q dist
    W_std = np.exp(W_rho)
    b_std = np.exp(b_rho)

    W = W_eps * W_std + W_mu
    b = b_eps * b_std + b_mu
    
    # Bayes by Backprop training
    W_kl = normal_kldiv(W_mu, 0., W_rho, 0.)
    b_kl = normal_kldiv(b_mu, 0., b_rho, 0.)
    W_kl, b_kl = np.sum(W_kl), np.sum(b_kl)

    kl_loss = W_kl + b_kl
    kl_loss = kl_loss + np.array(kl) 

    return (np.dot(inputs, W) + b, kl_loss)

  return init_fun, apply_fun

def Wrapper(layer):
  """
  layer: brax layer (e.g. Relu)
  """
  init_fn, apply_fn = layer
  def apply_fun(params, inputs, **kwargs):
    return (apply_fn(params, inputs[0], **kwargs), *inputs[1:])
  return init_fn, apply_fun

def SDEActWrapper(layer):
  """For transitioning from SDELayer outputs to Activation layers
  TODO: combine and clean this up with the below Wrapper
  """
  init_fn, apply_fn = layer
  def apply_fun(params, inputs, rng, **kwargs):
    preds, postw, postkl, priorx, priorw, priorkl = inputs
    preds = apply_fn(params, preds, **kwargs)
    return preds, postw, postkl, priorx, priorw, priorkl

  return init_fn, apply_fun

def SDEWrapper(layer):
  """For transitioning from SDELayer outputs to Dense layers
  """
  init_fn, apply_fn = layer
  def apply_fun(params, inputs, rng, **kwargs):
    preds, postw, postkl, priorx, priorw, priorkl = inputs
    preds, postkl = apply_fn(params, (preds, postkl), rng, **kwargs)
    return preds, postw, postkl, priorx, priorw, priorkl

  return init_fn, apply_fun

def BayesianLayer(layer):
  """
  layer: (init_fn, apply_fn)
  Make any layer Bayesian.
  """
  _init_fn, _apply_fn = layer
  def init_fun(rng, input_shape):
    k1, k2 = random.split(rng, 2)
    output_shape, W_mu = _init_fn(k2, input_shape) # mean of w distribution
    W_rho = tree_map(lambda w: np.ones_like(w) * np.log(np.exp(1e-3) - 1), W_mu) # inverse of softplus

    return output_shape, (W_mu, W_rho)
  
  def apply_fun(params, inputs, rng, **kwargs):
    W_mu, W_rho = params
    W_mu_flat, unravel_fn = ravel_pytree(W_mu)
    W_rho_flat, _ = ravel_pytree(W_rho)
    W_sigma_flat = jax.nn.softplus(W_rho_flat)
    W_eps = random.normal(rng, W_mu_flat.shape)

    W_flat = W_sigma_flat * W_eps + W_mu_flat
    W = unravel_fn(W_flat)
    output = _apply_fn(W, inputs, **kwargs)
    W_kl = np.sum(normal_kldiv(W_mu_flat, 0., np.log(W_sigma_flat), kwargs['logsigma2']))

    return output, W, W_kl

  return init_fun, apply_fun
