import torch
import torch.nn as nn

from . import bpq_cuda


class BallPointQueryFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, pcs, centroids, radius, max_samples):
        outputs = bpq_cuda.forward(pcs, centroids, radius, max_samples)
        group_idx = outputs[0].long()
        ctx.mark_non_differentiable(group_idx)
        return group_idx


class BallPointQuery(nn.Module):

    def __init__(self, radius, max_samples):
        super(BallPointQuery, self).__init__()
        self.radius = radius
        self.max_samples = max_samples

    def forward(self, pcs, centroids):
        return BallPointQueryFunction.apply(pcs, centroids, self.radius,
                                            self.max_samples)
