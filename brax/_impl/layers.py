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

def ConcatSquashLineartx(out_dim, W_init=he_normal(), b_init=normal()):
  """ y = Sigmoid(at + c)(Wx + b) + dt. Note: he_normal only takes multi dim.
  """
  def init_fun(rng, input_shape):
    output_shape = input_shape[:-1] + (out_dim,)
    k1, k2, k3, k4, k5 = random.split(rng, 5)
    W, b = W_init(k1, (input_shape[-1], out_dim)), b_init(k2, (out_dim,))
    w_t, w_tb = b_init(k3, (out_dim,)), b_init(k4, (out_dim,))
    b_t = b_init(k5, (out_dim,))
    return output_shape, (W, b, w_t, w_tb, b_t)

  def apply_fun(params, inputs, **kwargs):
    t, x = inputs
    W, b, w_t, w_tb, b_t = params

    # (W.xtt + b) *
    out = np.dot(x, W) + b
    # sigmoid(a.t + c)  + 
    out *= jax.nn.sigmoid(w_t * t + w_tb) 
    # d.t
    out += b_t * t

    return (t, out)
  return init_fun, apply_fun
        

def DiffEqWrappertx(layer):
  """Wrapper for time dependent layers
  """
  init_fun, layer_apply_fun = layer
  def apply_fun(params, inputs, **kwargs):
    t, x = inputs
    return t, layer_apply_fun(params, x, **kwargs)
  return init_fun, apply_fun

def augmented_post_drift(rng, state_unravel_fn, param_unravel_fn, driftx_apply_fn, prior_driftw_apply_fn, driftw0_apply_fn,
                          driftw_apply_fn, diff_apply_fn, W_driftwt, W_priorwt, W_diffusion, W0_post, setting, t, aug, eps, **kwargs): # last 3 partial
  """aug: [x, w, kl]
  """
  flat_w, x, kl = state_unravel_fn(aug) # flat_w inputs
  w = param_unravel_fn(flat_w) # hierarchical w params

  _, dxt = driftx_apply_fn(w, (t, x), **kwargs) # TODO: does this need to be bayesian? No we sample once only.

  _, dwtkl = driftw_apply_fn(W_driftwt, (t, flat_w), **kwargs)
  
  # pw_drift = 0. # assume prior has zero drift
  _, pw_drift = prior_driftw_apply_fn(W_priorwt, (t, flat_w), **kwargs) # account for prior
  _, diffwt = diff_apply_fn(W_diffusion, (t, flat_w), **kwargs) # with time dependence


  # stop gradient only to compute the KL
  _dwtkl = dwtkl # already "includes the w0kl", no need to add again.
  u = (_dwtkl - pw_drift) * 0
  if setting['diffw']:
    u = (_dwtkl - pw_drift) / diffwt
  dkl = np.sum((u ** 2) / 2) # dt = 1
  u2 = u * eps
  dkl_stl = np.sum(u2)

  # _w0kl, _w0_prior = 0, 0 # for gaussian mixture on W0
  if setting['stop_grad']:
    print("applying sticking the landing")
    if setting['priorw0_sigma'] >= 0:
      print("stopping gradient on w0 posterior")
      # stop grad on posterior over w0
      W0_post = tree_map(lambda w: lax.stop_gradient(w), W0_post)
      # _w0kl = tree_map(lambda w: lax.stop_gradient(w), _w0kl) # TODO: alternative?
      # w0out, _w0, _w0kl = driftw0_apply_fn(W0_post, (0, x), rng=rng, logsigma2=np.log(setting['priorw0_sigma']))
    # approx posterior over weights
    if setting['diff_drift']:
      print("stopping gradient on difference parameterization")
      W_driftwtkl = tree_map(lambda w: lax.stop_gradient(w), W_driftwt[-1])
      W_driftwtkl = (W_driftwt[0], W_driftwtkl) # first slot is params for t, which are ()
      _, _dwtkl = driftw_apply_fn(W_driftwtkl, (t, flat_w), **kwargs) # dwtkl
    else:
      W_driftwtkl = tree_map(lambda w: lax.stop_gradient(w), W_driftwt)
      _, _dwtkl = driftw_apply_fn(W_driftwtkl, (t, flat_w), **kwargs) # dwtkl

  # for original (but wrong) formulation of STL
  # u = (_dwtkl - pw_drift) * 0
  # if setting['diffw']:
  #   u = (_dwtkl - pw_drift) / diffwt
  # dkl = np.sum((u ** 2) / 2)
  ## dkl = dkl + _w0kl # account for bayesian w0 parameters

  # New STL: second term stop gradient (not first term - moved above)
  if setting['stop_grad']:
    print("stopping gradient on second STL term")
    u2 = (_dwtkl - pw_drift) * 0
    if setting['diffw']:
      u2 = (_dwtkl - pw_drift) / diffwt
    u2 = u2 * eps # u dBt
    dkl_stl = np.sum(u2)
  dkl = dkl + dkl_stl

  # outputs = [dxt, dwtkl, dkl]
  outputs = [dwtkl, dxt, dkl]

  return ravel_pytree(outputs)[0]

def augmented_prior_drift(state_unravel_fn, param_unravel_fn, driftx_apply_fn, prior_driftw_apply_fn, diff_apply_fn, 
                          W_priorwt, W_diffusion, setting, t, aug, eps, **kwargs): # partial last 3
  flat_w, x, kl = state_unravel_fn(aug)  # flat_w inputs
  w = param_unravel_fn(flat_w)  # hierarchical w params

  _, dxt = driftx_apply_fn(w, (t, x), **kwargs)
  _, dwtkl = prior_driftw_apply_fn(W_priorwt, (t, flat_w), **kwargs)

  pw_drift = 0.  # assume prior process has zero drift
  _, diffwt = diff_apply_fn(W_diffusion, (t, flat_w), **kwargs)  # with time dependence

  u = (dwtkl - pw_drift) * 0
  if setting['diffw']:
    u = (dwtkl - pw_drift) / diffwt
  dkl = np.sum((u ** 2) / 2)

  # new FMC formulation
  u2 = u * eps
  dkl_fmc = np.sum(u2)
  dkl = dkl + dkl_fmc

  # outputs = [dxt, dwtkl, dkl]
  outputs = [dwtkl, dxt, dkl]
  return ravel_pytree(outputs)[0]

def augmented_diffusion(state_unravel_fn, param_unravel_fn, diff_apply_fn, W_diffusion, t, aug, **kwargs): #partial last 3
  # (40, 1), (481,) = (40 + 1 + 481 = 522) for arch [1,20], [20,20], [20,1]
  flat_w, x, kl = state_unravel_fn(aug)
  # print("diff shapes", x.shape, flat_w.shape, kl.shape)
  w = param_unravel_fn(flat_w)

  gxt = np.zeros_like(x)
  _, gwt = diff_apply_fn(W_diffusion, (t, flat_w), **kwargs)
  gkl = np.zeros_like(kl)

  # outputs = [gxt, gwt, gkl] # (40, 1) () ()
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
    _, W0 = driftx_init_fn(k1, input_shape)
    _, W_driftwt = driftw_init_fn(k2, (-1, w_dim))
    _, W_difft = diff_init_fn(k3, (-1, w_dim))
    _, W_priorwt = prior_driftw_init_fn(k4, (-1, w_dim))

    print(f"scaling drift by {setting['driftw_scale']}")
    W_driftwt = tree_map(lambda w: w * setting['driftw_scale'], W_driftwt)
    
    print("initialized last layer params to 0") # done in mlp definition

    if driftw0_layer:
      print("calculating W_mu and W_rho for W0")
      k5, k6 = random.split(rng)
      _, W0_post = driftw0_init_fn(k5, input_shape)
      if setting['priorw0_sigma'] >= 0:
        print(f"priorw0_sigma set to he normal stdev {setting['priorw0_sigma']}")
        # set the prior distribution to be same as he initialization stdev
        setting['priorw0_sigma'] = he_normal_stdev(k6, (input_shape[-1], 1)) # TODO: correct shape?
        with open(os.path.join(setting['save_dir'], "priorw0.txt"), 'a') as f:
          f.write(setting['priorw0_sigma'])
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

    # _, W0 = driftx_init_fn(rng, (-1, x_dim)) # W0 is randomized each call
    # Bayesian about W0 (mean field)
    W0kl = 0.
    if W0_post is not None:
      tw0 = 0 # NOTE: this is just a dummy variable to be compatible with `shape_dependent`
      _, W0, W0kl = driftw0_apply_fn(W0_post, (tw0, inputs), rng=rng, logsigma2=np.log(setting['priorw0_sigma']))
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
      # outputs = sdeint(aug_post_drift, aug_diffusion, aug_init_(inputs), np.linspace(0, 1, 100), rng)
      outputs = sdeint(aug_post_drift, aug_diffusion, aug_init_(inputs)[0], aug_init_(inputs)[1], np.linspace(0, 1, 90), rng) # TODO: to prevent oom problems
      ws, xs, kls = zip(*[state_unravel_fn(out) for out in outputs]) # list(x_dim), list(w_dim), list(1)
      # prior_outputs = sdeint(aug_prior_drift, aug_diffusion, aug_init_(inputs), np.linspace(0, 1, 100), rng)
      prior_outputs = sdeint(aug_prior_drift, aug_diffusion, aug_init_(inputs)[0], aug_init_(inputs)[1], np.linspace(0, 1, 90), rng) # TODO: 100 steps
      prior_ws, prior_xs, prior_kls = zip(*[state_unravel_fn(out) for out in prior_outputs]) # list(x_dim), list(w_dim), list(1)
      return xs, ws, kls, prior_xs, prior_ws, prior_kls

    solution = _sdeint(aug_post_drift, aug_diffusion, aug_init_(inputs)[0], aug_init_(inputs)[1], np.linspace(0, 1, 20), rng) # was sdeint[-1]
    w, x, kl = state_unravel_fn(solution)
    prior_outputs = _sdeint(aug_prior_drift, aug_diffusion, aug_init_(inputs)[0], aug_init_(inputs)[1], np.linspace(0, 1, 20), rng) # TODO: ramp up 100 steps for better performance
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
    # TODO: don't use diffeq wrapper, get t, x = inputs??
    params1, params2 = params
    t1, out1 = apply_fn1(params1, inputs, **kwargs)
    t2, out2 = apply_fn2(params2, inputs, **kwargs)
    # t1 == t2 since sdeint calls
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
  # swish = lambda beta, x: x * jax.nn.sigmoid(x * jax.nn.softplus(beta)) # no / 1.1 for lipschitz

  def init_fun(rng, input_shape):
    output_shape = input_shape[:-1] + (out_dim,)
    beta0 = np.ones((out_dim,)) * beta_init
    return input_shape, (np.array(beta0),)

  def apply_fun(params, inputs, **kwargs):
    beta_, = params
    ret = swish(beta_, inputs)
    return ret
  
  return init_fun, apply_fun

# passable_swish = elementwise(Swish(out_dim)[1])


def Flatten():
  """
  Flattens all but leading dim in input of convolutional layer e.g. (100, 28, 28, 1) -> (100, 784)
  """
  def init_fun(rng, input_shape):
    output_shape = (input_shape[0], functools.reduce(op.mul, input_shape[1:], 1))
    return output_shape, ()

  def apply_fun(params, inputs, **kwargs):
    # t, x = inputs # NOTE: get the same effect as wrapping this inside diffeqWrapper?
    x = inputs
    x_flat = x.reshape((x.shape[0], -1))
    print(f"flattened from {x.shape} to {x_flat.shape}")
    # return (t, x_flat)
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
    # zeros_ = np.zeros((inputs.shape[0], 1 * pad_zeros)) # hardcoded for 1D case
    # xzeros = np.concatenate([x, zeros_], axis=-1)
    # print(f"augmented from {inputs.shape} to {xzeros.shape}")
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
    # print(inputs[0][0])
    inputs, kl = inputs
    # kl = 0
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
    kl_loss = kl_loss + np.array(kl) # TODO: why do we get compatibility issues?
    # print(W.shape)

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
    W_eps = random.normal(rng, W_mu_flat.shape) # NOTE: sampling from a Gaussian

    W_flat = W_sigma_flat * W_eps + W_mu_flat
    W = unravel_fn(W_flat)
    output = _apply_fn(W, inputs, **kwargs)
    # W_kl = np.sum(normal_kldiv(W_mu_flat, 0., np.log(W_sigma_flat), 0.))
    W_kl = np.sum(normal_kldiv(W_mu_flat, 0., np.log(W_sigma_flat), kwargs['logsigma2']))

    return output, W, W_kl # TODO: in a normal Bayesian layer, should just be `output, W_kl`

  return init_fun, apply_fun