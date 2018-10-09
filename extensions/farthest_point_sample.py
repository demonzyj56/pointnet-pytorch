import torch
import torch.nn as nn

from . import fps_cuda


class FarthestPointSampleFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, pcs, num_centroids):
        out = torch.zeros(pcs.size(0), num_centroids, dtype=torch.int64,
                          device=pcs.device)
        fps_cuda.forward(pcs, out)
        ctx.mark_non_differentiable(out)
        return out


class FarthestPointSample(nn.Module):

    def __init__(self, num_centroids):
        super(FarthestPointSample, self).__init__()
        self.num_centroids = num_centroids

    def forward(self, pcs):
        assert pcs.size(-1) >= self.num_centroids
        return FarthestPointSampleFunction.apply(pcs, self.num_centroids)
