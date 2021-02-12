# Infinitely Deep Bayesian Neural Networks with SDEs

This library contains JAX and Pytorch implementations of neural ODEs and Bayesian layers for stochastic variational inference. 
A rudimentary JAX implementation of differentiable SDE solvers is also provided, refer to [torchsde](https://github.com/google-research/torchsde) 
for a full set of differentiable SDE solvers in Pytorch.

## Installation
To run code, execute:
```
pip install -r requirements.txt
```
_Note_: Package version may change, refer to each package's official page for installation instructions.

## JaxSDE: Differentiable SDE Solvers in JAX
The `jaxsde` library contains SDE solvers in the Ito and Stratonovich form. 
Different solvers can be specified with the following `method={euler_maruyama|milstein|euler_heun}`. 
Stochastic adjoint (`sdeint_ito`) training mode does not work efficiently yet, use `sdeint_ito_fixed_grid` for now.

### Usage
Default solver:
Backpropagation through the solver.
```
from jaxsde.jaxsde.sdeint import sdeint_ito_fixed_grid

y1 = sdeint_ito_fixed_grid(f, g, y0, ts, rng, fw_params, method="euler_maruyama")
```

Stochastic adjoint:
Using O(1) memory instead of solving an adjoint SDE in the backward pass.
```
from jaxsde.jaxsde.sdeint import sdeint_ito

y1 = sdeint_ito(f, g, y0, ts, rng, fw_params, method="milstein")
```

## Brax: Bayesian SDE Framework in JAX
Implementation of composable Bayesian layers in the [stax](https://jax.readthedocs.io/en/latest/jax.experimental.stax.html) API.
Our SDE Bayesian layers can be used with the `SDEBNN` block composed with multiple parameterizations of time-dependent layers in `diffeq_layers`.
Sticking-the-landing (STL) trick can be enabled during training with `--stl` for improving convergence rate.
Augment the inputs by a custom amount `--aug <integer>`, set the number of samples averaged over with `--nsamples <integer>`.
If memory constraints pose a problem, train in gradient accumulation mode: `--acc_grad` and gradient checkpointing: `--remat`. 

### Usage
All examples can be swapped in with different vision datasets and includes tensorboard logging for critical metrics.

#### Toy 1D regression to learn complex posteriors:
```
python examples/jax/sdebnn_toy1d.py
```

#### Image Classification:
To train an SDEBNN model:
```
python examples/jax/sdebnn_classification.py --output <output directory> --model sdenet --aug 2 --nblocks 2-2-2 --diff_coef 0.2 --fx_dim 64 --fw_dims 2-64-2 --nsteps 20 --nsamples 1
```
To train a ResNet baseline, specify `--model resnet` and for a Bayesian ResNet baseline, specify `--meanfield_sdebnn`.

## Torchsde: SDE-BNN in Pytorch
A PyTorch implementation of the Brax framework powered by the [torchsde](https://github.com/google-research/torchsde) backend.

### Usage
All examples can be swapped in with different vision datasets and includes tensorboard logging for critical metrics.

#### Toy 1D regression to learn multi-modal posterior:
```
python examples/torch/sdebnn_toy1d.py --output_dir <dst_path> --ds b40gap --diff_const 0.2 --prior_dw ou --num_samples 100 --stl
```

#### Image Classification:
All hyperparameters can be found in the training script.
```
python examples/torch/sdebnn_classification.py --train-dir <output directory> --data cifar10 --dt 0.05 --method midpoint --adjoint True
--inhomogeneous True
```

## References
TBA
