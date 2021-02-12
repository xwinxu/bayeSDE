import pickle

import jax
import jax.numpy as jnp
import jax.random
import numpy as np
import sklearn.metrics as metrics
from jax import lax
from sklearn.metrics import log_loss


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
    for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def jax_rng_generator(seed):
    rng = jax.random.PRNGKey(seed)
    while True:
        rng, rng_out = jax.random.split(rng, 2)
        yield rng_out

class jaxRNG:
    def __init__(self, seed):
        self.rng = jax.random.PRNGKey(seed)
    def next(self):
        # while True:
        self.rng, rng_out = jax.random.split(self.rng, 2)
        return rng_out


def save_params(_params, save_dir, pkl=False):
    # save_dir should include full path including filename
    if pkl:
        with open(save_dir, "wb") as f:
            pickle.dump(_params, f)
        # print(f"pickled params to {save_dir}")
    else:
        _params = jnp.stack(_params, axis=0)
        np.save(save_dir, np.array(_params))
        # print(f"np saved to {save_dir}")


def load_params(save_dir, unravel_fn, pkl=False):
    if pkl:
        state = {}
        print("loading pickled params...")
        with open(save_dir, "rb") as f:
            state = pickle.load(f)
        print(f"loaded pickled params: {state}")
        return unravel_fn(state)
    state = np.load(save_dir)
    print(f"loaded .npy from {save_dir}")
    return state


def get_lr_schedule(sched, nb, lr, warmup=1e3, decay_steps=19500, decay_rate=0.1):

    def _exponential(lr):
        return jax.experimental.optimizers.exponential_decay(lr, decay_steps, decay_rate)

    def _cosine(lr, alpha=0.1):
        def _cosine_sched(itr):
            step = jnp.minimum(itr, decay_steps)
            cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * step / decay_steps))
            decayed = (1 - alpha) * cosine_decay + alpha
            return lr * decayed
        return _cosine_sched

    def _inv(lr, decay_steps=19500, staircase=False):
        return jax.experimental.optimizers.inverse_time_decay(lr, decay_steps,
                                                            decay_rate, staircase)

    def _stair(lr):
        def _step_decay(itr):
            lrate = lr * jax.lax.pow(decay_rate, jax.lax.floor((1 + itr)/decay_steps))
            return lrate
        return _step_decay

    def _custom(lr):
        # time based decay
        def custom_lr(itr):
            _epoch = itr // nb
            id = lambda x: x
            return lax.cond(_epoch < 40, 9e-4, id, 0,
                            lambda _: lax.cond(_epoch < 150, 5e-4, id, 0,
                              lambda _: lax.cond(_epoch < 210, 1e-4, id, 1e-5, id)))
        return custom_lr

    def _custom2(lr):
        # time based decay
        def custom_lr2(itr):
            _epoch = itr // nb
            id = lambda x: x
            return lax.cond(_epoch < 40, 9e-4, id, 0,
                            lambda _: lax.cond(_epoch < 100, 7e-4, id, 0,
                              lambda _: lax.cond(_epoch < 200, 5e-4, id, 1e-4, id)))
        return custom_lr2

    def _warmup(lr):
        # guessed warmup schedule
        def _warmup_sched(itr):
            itr_frac = lax.min((itr.astype(jnp.float32) + 1.) / lax.max(warmup, 1.), 1.)
            _epoch = itr // nb
            id = lambda x: x
            return lax.cond(_epoch < 55, lr * itr_frac, id, lr / 10, id)
        return _warmup_sched

    _schedules = {'exp': _exponential, 'cos': _cosine, 'stair': _stair,
                'inv': _inv, 'custom': _custom, 'custom2': _custom2, 'warmup': _warmup}

    return _schedules[sched](lr)


def get_calibration(y, p_mean, num_bins=10):
    """Compute the calibration.
    References:
    https://arxiv.org/abs/1706.04599
    https://arxiv.org/abs/1807.00263
    Args:
      y: one-hot encoding of the true classes, size (?, num_classes)
      p_mean: numpy array, size (batch_size, num_classes)
            containing the mean output predicted probabilities
      num_bins: number of bins
    Returns:
      cal: a dictionary
        {reliability_diag: realibility diagram
        ece: Expected Calibration Error
        mce: Maximum Calibration Error
        }
    """
    # Compute for every test sample x, the predicted class.
    class_pred = np.argmax(p_mean, axis=1) 
    # and the confidence (probability) associated with it.
    conf = np.max(p_mean, axis=1)
    # Convert y from one-hot encoding to the number of the class
    y = np.argmax(y, axis=1)
    # Storage
    acc_tab = np.zeros(num_bins)  # empirical (true) confidence
    mean_conf = np.zeros(num_bins)  # predicted confidence
    nb_items_bin = np.zeros(num_bins)  # number of items in the bins
    tau_tab = np.linspace(0, 1, num_bins+1)  # confidence bins (11,)
    for i in np.arange(num_bins):  # iterate over the bins
        # select the items where the predicted max probability falls in the bin
        sec = (tau_tab[i + 1] > conf) & (conf >= tau_tab[i]) # (128,)
        nb_items_bin[i] = np.sum(sec)
        # select the predicted classes, and the true classes
        class_pred_sec, y_sec = class_pred[sec], y[sec]
        # average of the predicted max probabilities
        mean_conf_update = np.mean(conf[sec]) if nb_items_bin[i] > 0 else np.nan
        mean_conf[i] = mean_conf_update
        # compute the empirical confidence
        acc_tab_update = np.mean(class_pred_sec == y_sec) if nb_items_bin[i] > 0 else np.nan
        acc_tab[i] = acc_tab_update

    # Reliability diagram (this is moved up so clean up does not change the dimension)
    reliability_diag = (mean_conf, acc_tab)

    # Cleaning
    mean_conf = mean_conf[nb_items_bin > 0]
    acc_tab = acc_tab[nb_items_bin > 0]
    final_nb_items_bin = nb_items_bin[nb_items_bin > 0]

    # Expected Calibration Error
    _weights = 0.
    ece = np.sum(_weights * np.absolute(mean_conf - acc_tab))
    if not np.sum(final_nb_items_bin) == 0:
        print(f"there were {nb_items_bin} items in bins 1-{num_bins}")
        _weights = final_nb_items_bin.astype(np.float32) / np.sum(final_nb_items_bin)
        ece = np.average(
            np.absolute(mean_conf - acc_tab),
            weights=_weights)
    # Maximum Calibration Error
    # mce = np.max(np.absolute(mean_conf - acc_tab)) # this gives np.max(empty) errors
    cal = {'reliability_diag': reliability_diag,
          'ece': ece,
          'nb_items': nb_items_bin/np.sum(nb_items_bin)
          }
          # 'mce': mce}
    return cal

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


def score_model(probs, y_true, verbose=False, normalize=False, bins=15):
    """
    Evaluate model using various scoring measures: Error Rate, ECE, MCE, NLL, Brier Score

    Params:
        probs: a list containing probabilities for all the classes with a shape of (samples, classes)
        y_true: a list containing the actual class labels
        verbose: (bool) are the scores printed out. (default = False)
        normalize: (bool) in case of 1-vs-K calibration, the probabilities need to be normalized.
        bins: (int) - into how many bins are probabilities divided (default = 15)

    Returns:
        (error, ece, mce, loss, brier), returns scores dictionary
    """
    preds = np.argmax(probs, axis=1)  # Take maximum confidence as prediction
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
    probs[nan_mask] = 0.
    y_pred = probs
    _loss = log_loss(y_true=y_true, y_pred=y_pred)
    
    y_prob_true = np.array([y_pred[i, idx] for i, idx in enumerate(y_true)])  # Probability of positive class
    brier = brier_score(y_true, y_prob_true)  # Brier Score (multiclass)

    if verbose:
        print("Accuracy:", accuracy)
        print("Error:", error)
        print("ECE:", ece)
        print("MCE:", mce)
        print("Loss:", _loss)
        print("brier:", brier)

    scores = {'acc': accuracy, 'error': error, 'ece': ece, 'mce': mce, 'brier': brier, 'loss': _loss}

    return (accuracy, error, ece, mce, brier, _loss)

def check_nans(info_array):
    checks = jax.tree_util.tree_map(lambda x: jax.lax.is_finite(jax.lax.convert_element_type(x, jnp.float32)), info_array)
    all_true = jnp.all(jnp.array(checks))
    return bool(all_true)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    # alist.sort(key=natural_keys) sorts in human order
    # http://nedbatchelder.com/blog/200712/human_sorting.html
    import re
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def _compute_fans(shape, in_axis=-2, out_axis=-1):
  receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
  fan_in = shape[in_axis] * receptive_field_size
  fan_out = shape[out_axis] * receptive_field_size
  return fan_in, fan_out

def variance_scaling(scale, mode, distribution, in_axis=-2, out_axis=-1, dtype=np.float32):
  from jax import random
  def init(key, shape, dtype=dtype):
    fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
    if mode == "fan_in": denominator = fan_in
    else:
      raise ValueError("variance scaling requires he_normal mode for now, not provided mode {}".format(mode))

    variance = jnp.array(scale / denominator, dtype=dtype)
    if distribution == "truncated_normal":
      # constant is stddev of standard normal truncated to (-2, 2)
      stddev = jnp.sqrt(variance) / jnp.array(.87962566103423978, dtype)
      return random.truncated_normal(key, -2, 2, shape, dtype) * stddev, stddev
    else:
      raise ValueError("invalid distribution for variance scaling initializer")
  return init

def he_normal_stdev(rng, input_shape):
  init_fn = variance_scaling(2.0, "fan_in", "truncated_normal")
  return init_fn(rng, input_shape)[1]
