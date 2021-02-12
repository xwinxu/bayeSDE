_ACTIVATIONS = dict()
_LOSSES = dict()
_DATA = dict()


def add_data(dataset):
  def add_to_data(fn):
    _DATA[dataset] = fn
    return fn
  return add_to_data

def get_data(ds_):
  if ds_ not in _DATA:
    print(f"Error dataset {ds_} not found")
    exit()
  return _DATA[ds_]


def add_loss(loss):
  def add_to_loss(fn):
    _LOSSES[loss] = fn
    return fn
  return add_to_loss

def get_loss(loss_):
  if loss_ not in _LOSSES:
    print(f"Error loss function {loss_} not found")
    exit()
  return _LOSSES[loss_]


def register(name):

  def add_to_dict(fn):
    global _ACTIVATIONS
    _ACTIVATIONS[name] = fn
    return fn

  return add_to_dict


def get_activn(hparams):
  if hparams not in _ACTIVATIONS:
    print(f'Error activation function {hparams} not found')
    exit()
  return _ACTIVATIONS[hparams]
