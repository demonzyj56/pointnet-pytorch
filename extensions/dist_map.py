import torch
import torch.nn as nn

from . import dist_map_cuda


class DistMapFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, b):
        outputs = dist_map_cuda.forward(a, b)
        ctx.save_for_backward(a, b)

        return outputs[0]

    @staticmethod
    def backward(ctx, grad_d):
        outputs = dist_map_cuda.backward(grad_d.contiguous(),
                                         *ctx.saved_variables)
        d_a, d_b = outputs
        return d_a, d_b


class DistMap(nn.Module):

    def __init__(self):
        super(DistMap, self).__init__()

    def forward(self, a, b):
        return DistMapFunction.apply(a, b)
