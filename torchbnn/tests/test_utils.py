import torch
from torchbnn._impl import utils

l = [[torch.randn(5, 2), torch.randn(3, 2)], torch.randn(5, 2),
     [torch.randn(3, 1), [torch.randn(3, 2), torch.randn(())]]]
flat, unravel = utils.ravel_pytree(l)

ul = unravel(flat)
print(ul)
print(l)
