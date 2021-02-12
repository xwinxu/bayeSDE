from functools import partial

import jax.numpy as np
from jax import lax, random
from jax.experimental import stax
from jax.experimental.stax import (AvgPool, BatchNorm, FanInSum, FanOut,
                                   Flatten, Identity)
from jax.nn.initializers import ones, variance_scaling, zeros

from brax._impl.conv import Conv

__all__ = [
    # ResNet for CIFAR.
    "CifarResNet",
    "resnet20",
    "resnet32",
    "resnet44",
    "resnet56",
    "resnet110",
    "resnet1202",
    # ResNet for ImageNet.
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "wide_resnet_18_2",
    "wide_resnet_34_2",
]


def GroupNorm(num_groups=16, group_size=None, epsilon=1e-6, bias=True,
              scale=True, bias_init=zeros, scale_init=ones):
    if ((num_groups is None and group_size is None) or
            (num_groups is not None and group_size is not None)):
        raise ValueError('Either `num_groups` or `group_size` should be '
                         'specified, but not both of them.')
    _bias_init = lambda rng, shape: bias_init(rng, shape) if bias else ()
    _scale_init = lambda rng, shape: scale_init(rng, shape) if scale else ()

    def init_fun(rng, input_shape):
        shape = tuple([1 for d in input_shape[:-1]] + [input_shape[-1]])
        k1, k2 = random.split(rng)
        bias, scale = _bias_init(k1, shape), _scale_init(k2, shape)
        return input_shape, (bias, scale)

    def apply_fun(params, x, **kwargs):
        bias_value, scale_value = params

        groups_num = num_groups
        if group_size is not None:
            channels = x.shape[-1]
            if channels % group_size != 0:
                raise ValueError('Number of channels ({}) is not multiple of the '
                                 'group size ({}).'.format(channels, group_size))
            groups_num = channels // groups_num

        input_shape = x.shape
        group_shape = x.shape[:-1] + (groups_num, x.shape[-1] // groups_num)

        x = x.reshape(group_shape)

        reduction_axis = [d for d in range(1, x.ndim - 2)] + [x.ndim - 1]

        mean = np.mean(x, axis=reduction_axis, keepdims=True)
        mean_of_squares = np.mean(np.square(x), axis=reduction_axis,
                                  keepdims=True)
        var = mean_of_squares - np.square(mean)

        x = (x - mean) * lax.rsqrt(var + epsilon)

        x = x.reshape(input_shape)

        if scale and bias:
            return x * scale_value + bias_value
        if scale:
            return x * scale_value
        if bias:
            return x + bias_value
    return init_fun, apply_fun


def fixup_init(num_layers):
    def init(key, shape, dtype=np.float32):
        scale = np.sqrt(2 / (shape[0] * np.prod(shape[2:]))) * num_layers ** (-0.5)
        return variance_scaling(
            scale=scale, mode="fan_in", distribution="normal")(key, shape, dtype)
    return init


def maybe_use_normalization(normalization_method=None):
    if normalization_method == "batch_norm":
        return BatchNorm()
    elif normalization_method == "group_norm":
        return GroupNorm()
    else:
        return Identity


def LambdaLayer(lambda_fn):
    def init_fun(rng, input_shape):
        x = np.zeros(tuple(i if i != -1 else 1 for i in input_shape))
        y = lambda_fn(x)
        return (y.shape, ())

    apply_fun = lambda params, inputs, **kwargs: lambda_fn(inputs)
    return init_fun, apply_fun


def _shortcut_pad(x):
    if x.ndim == 4:
        x = x[:, ::2, ::2, :]
    elif x.ndim == 3:
        x = x[::2, ::2, :]
    z = np.zeros_like(x)
    return np.concatenate((x, z), axis=-1)


def AvgPoolAll():
    def init_fun(rng, input_shape):
        out_shape = list(input_shape)
        out_shape[-2] = 1
        out_shape[-3] = 1
        return out_shape, ()

    def apply_fun(params, x, **kwargs):
        out = np.sum(x, axis=(-2, -3), keepdims=True) / (x.shape[-1] * x.shape[-2])
        return out

    return init_fun, apply_fun


def FixupBias():
    init_fun = lambda rng, input_shape: (input_shape, np.zeros(1))
    apply_fun = lambda params, inputs, **kwargs: inputs + params
    return init_fun, apply_fun


def FixupScale():
    init_fun = lambda rng, input_shape: (input_shape, np.ones(1))
    apply_fun = lambda params, inputs, **kwargs: inputs * params
    return init_fun, apply_fun


def CifarBasicBlock(planes, stride=1, option="A", normalization_method=None,
                    use_fixup=False, num_layers=None, w_init=None, actfn=stax.Relu):
    Main = stax.serial(
        FixupBias() if use_fixup else Identity,
        Conv(planes, (3, 3), strides=(stride, stride), padding="SAME",
             W_init=fixup_init(num_layers) if use_fixup else w_init,
             bias=False),
        maybe_use_normalization(normalization_method),
        FixupBias() if use_fixup else Identity,
        actfn,
        FixupBias() if use_fixup else Identity,
        Conv(planes, (3, 3), padding="SAME", bias=False,
             W_init=zeros if use_fixup else w_init),
        maybe_use_normalization(normalization_method),
        FixupScale() if use_fixup else Identity,
        FixupBias() if use_fixup else Identity,
    )
    Shortcut = Identity
    if stride > 1:
        if option == "A":
            # For CIFAR10 ResNet paper uses option A.
            Shortcut = stax.serial(
                #  FixupBiast() if use_fixup else Identity,
                LambdaLayer(_shortcut_pad))
        elif option == "B":
            Shortcut = stax.serial(FixupBias() if use_fixup else Identity,
                                   Conv(planes, (1, 1), strides=(stride, stride),
                                        W_init=w_init, bias=False),
                                   maybe_use_normalization(normalization_method))
    return stax.serial(FanOut(2), stax.parallel(Main, Shortcut), FanInSum, actfn)


def CifarBasicBlockv2(planes, stride=1, option="A", normalization_method=None,
                      use_fixup=False, num_layers=None, w_init=None, actfn=stax.Relu):
    assert not use_fixup, "nah"
    Main = stax.serial(
        maybe_use_normalization(normalization_method),
        actfn,
        Conv(planes, (3, 3), strides=(stride, stride), padding="SAME",
             W_init=w_init, bias=False),
        maybe_use_normalization(normalization_method),
        actfn,
        Conv(planes, (3, 3), padding="SAME", W_init=w_init, bias=False),
    )
    Shortcut = Identity
    if stride > 1:
        if option == "A":
            # For CIFAR10 ResNet paper uses option A.
            Shortcut = LambdaLayer(_shortcut_pad)
        elif option == "B":
            Shortcut = Conv(planes, (1, 1), strides=(stride, stride),
                            W_init=w_init, bias=False)
    return stax.serial(FanOut(2), stax.parallel(Main, Shortcut), FanInSum)


def CifarResNet(block, num_blocks, expansion=1, option="A",
                normalization_method=None, use_fixup=False,
                init=None, actfn=stax.Relu):
    w_init = None
    if init == "he":
        w_init = partial(variance_scaling, 2.0, "fan_out", "truncated_normal")()
    num_layers = sum(num_blocks)

    def _make_layer(block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(planes, stride, option=option,
                                normalization_method=normalization_method,
                                use_fixup=use_fixup, num_layers=num_layers,
                                w_init=w_init, actfn=actfn))
        return stax.serial(*layers)

    return [
        Conv(16 * expansion, (3, 3), padding="SAME", W_init=w_init, bias=False),
        maybe_use_normalization(normalization_method),
        FixupBias() if use_fixup else Identity,
        actfn,
        _make_layer(block, 16 * expansion, num_blocks[0], stride=1),
        _make_layer(block, 32 * expansion, num_blocks[1], stride=2),
        _make_layer(block, 64 * expansion, num_blocks[2], stride=2),
        AvgPoolAll(),
        Flatten,
        FixupBias() if use_fixup else Identity,
    ]


def BasicBlock(planes, stride=1, downsample=None, base_width=64,
               norm_layer=Identity, actfn=stax.Relu):
    if base_width != 64:
        raise ValueError("BasicBlock only supports base_width=64")
    Main = stax.serial(
        Conv(planes, (3, 3), strides=(stride, stride), padding="SAME", bias=False),
        norm_layer,
        actfn,
        Conv(planes, (3, 3), padding="SAME", bias=False),
        norm_layer,
    )
    Shortcut = downsample if downsample is not None else Identity
    return stax.serial(FanOut(2), stax.parallel(Main, Shortcut), FanInSum, actfn)


def BottleneckBlock(planes, stride=1, downsample=None, base_width=64,
                    norm_layer=Identity, actfn=stax.Relu):
    width = int(planes * (base_width / 64.))
    Main = stax.serial(
        Conv(width, (1, 1), bias=False),
        norm_layer,
        actfn,
        Conv(width, (3, 3), strides=(stride, stride), padding="SAME", bias=False),
        norm_layer,
        actfn,
        Conv(planes * 4, (1, 1), bias=False),
        norm_layer,
    )
    Shortcut = downsample if downsample is not None else Identity
    return stax.serial(FanOut(2), stax.parallel(Main, Shortcut), FanInSum, actfn)


def ResNet(block, expansion, layers, normalization_method=None, width_per_group=64, actfn=stax.Relu):
    norm_layer = Identity
    if normalization_method == "group_norm":
        norm_layer = GroupNorm(32)
    elif normalization_method == "batch_norm":
        norm_layer = BatchNorm()
    base_width = width_per_group

    def _make_layer(block, planes, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = stax.serial(
                Conv(planes * expansion, (1, 1), strides=(stride, stride), bias=False),
                norm_layer,
            )
        layers = []
        layers.append(block(planes, stride, downsample, base_width, norm_layer))
        for _ in range(1, blocks):
            layers.append(block(planes, base_width=base_width,
                                norm_layer=norm_layer, actfn=actfn))
        return stax.serial(*layers)

    return [
        Conv(64, (3, 3), strides=(1, 1), padding="SAME", bias=False),
        norm_layer,
        actfn,
        # MaxPool((3, 3), strides=(2, 2), padding="SAME"),
        _make_layer(block, 64, layers[0]),
        _make_layer(block, 128, layers[1], stride=2),
        _make_layer(block, 256, layers[2], stride=2),
        _make_layer(block, 512, layers[3], stride=2),
        AvgPool((4, 4)),
        Flatten,
    ]


def resnet8(**kwargs):
    return CifarResNet(CifarBasicBlock, [1, 1, 1], **kwargs)


def resnet14(**kwargs):
    return CifarResNet(CifarBasicBlock, [2, 2, 2], **kwargs)


def resnet20(**kwargs):
    return CifarResNet(CifarBasicBlock, [3, 3, 3], **kwargs)


def resnet32(**kwargs):
    return CifarResNet(CifarBasicBlock, [5, 5, 5], **kwargs)


def resnet32v2(**kwargs):
    return CifarResNet(CifarBasicBlockv2, [5, 5, 5], **kwargs)


def resnet44(**kwargs):
    return CifarResNet(CifarBasicBlock, [7, 7, 7], **kwargs)


def resnet56(**kwargs):
    return CifarResNet(CifarBasicBlock, [9, 9, 9], **kwargs)


def resnet110(**kwargs):
    return CifarResNet(CifarBasicBlock, [18, 18, 18], **kwargs)


def resnet1202(**kwargs):
    return CifarResNet(CifarBasicBlock, [200, 200, 200], **kwargs)


def resnet18(**kwargs):
    return ResNet(BasicBlock, 1, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, 1, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(BottleneckBlock, 4, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(BottleneckBlock, 4, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):
    return ResNet(BottleneckBlock, 4, [3, 8, 36, 3], **kwargs)


def wide_resnet_18_2(**kwargs):
    kwargs["width_per_group"] = 64 * 2
    return ResNet(BasicBlock, 1, [2, 2, 2, 2], **kwargs)


def wide_resnet_34_2(**kwargs):
    kwargs["width_per_group"] = 64 * 2
    return ResNet(BasicBlock, 1, [3, 4, 6, 3], **kwargs)


def test(net):

    from jax.tree_util import tree_flatten

    rng = random.PRNGKey(0)

    init_params_fn, apply_fn = net
    _, params = init_params_fn(rng, (-1, 32, 32, 3))

    leafs, tree = tree_flatten(params)
    # print([leaf.shape for leaf in leafs])
    total_params = sum([np.prod(p.shape) for p in leafs])
    print("Total number of params", total_params)

    x = random.normal(rng, (1, 32, 32, 3))
    y = apply_fn(params, x)
    print(x.shape, y.shape)


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith("resnet"):
            print(net_name)
            test(globals()[net_name]())
            print()
