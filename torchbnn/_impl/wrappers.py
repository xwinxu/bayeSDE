from inspect import signature

import torch.nn as nn

__all__ = ["diffeq_wrapper", "reshape_wrapper"]


class DiffEqWrapper(nn.Module):
    def __init__(self, module):
        super(DiffEqWrapper, self).__init__()
        self.module = module

    def forward(self, t, y):
        if "t" in signature(self.module.forward).parameters:
            return self.module.forward(t, y)
        elif "y" in signature(self.module.forward).parameters:
            return self.module.forward(y)
        else:
            raise ValueError("Differential equation needs to either take (t, y) or (y,) as input.")

    def __repr__(self):
        return self.module.__repr__()


def diffeq_wrapper(layer):
    return DiffEqWrapper(layer)


class ReshapeDiffEq(nn.Module):
    def __init__(self, input_shape, net):
        super(ReshapeDiffEq, self).__init__()
        assert len(signature(net.forward).parameters) == 2, "use diffeq_wrapper before reshape_wrapper."
        self.input_shape = input_shape
        self.net = net

    def forward(self, t, x):
        batchsize = x.shape[0]
        x = x.view(batchsize, *self.input_shape)
        return self.net(t, x).view(batchsize, -1)

    def __repr__(self):
        return self.diffeq.__repr__()


def reshape_wrapper(input_shape, layer):
    return ReshapeDiffEq(input_shape, layer)
