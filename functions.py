"""Custom functions."""
import logging
import torch

logger = logging.getLogger(__name__)


class FarthestPointSampleFunction(torch.autograd.Function):
    """Functional form of farthest point sampling. Notice that no gradients
    are computed.
    WARNING: naive implementation!
    """

    @staticmethod
    def forward(ctx, xyz, num_centroids):
        """
        :param ctx:
            Context manager.
        :param xyz: [batch_size, 3, num_points]
            Input point cloud coordinates.
        :param num_centroids: int
            Number of centroids to sample at each batch location.
        :return:
            centroids: torch.LongTensor [batch_size, num_centroids]
                The centroids sampled.
        """
        length = xyz.pow(2).sum(dim=1, keepdim=True)
        # (batch_size, num_points, num_points)
        dists = xyz.permute(0, 2, 1).bmm(xyz).mul(-2).add(length).add(length.permute(0, 2, 1))
        dists.clamp_(min=0)
        out = torch.zeros((xyz.size(0), num_centroids), dtype=torch.int64,
                          device=xyz.device)
        out[:, 0] = torch.randint(low=0, high=xyz.size(-1), size=(xyz.size(0),),
                                  dtype=out.dtype, device=out.device)
        for i in range(xyz.size(0)):
            for j in range(1, num_centroids):
                cur = out[i, j-1].item()
                dists[i, :, cur] = -1.
                out[i, j] = dists[i, cur].argmax()
        ctx.mark_non_differentiable(out)
        return out


class BallPointQueryFunction(torch.autograd.Function):
    """Functional implementation of ball point query."""

    @staticmethod
    def forward(ctx, pcs, centroids, radius, max_samples):
        """
        When the points around the centroids within the radius are fewer than
        max_sample, assign repeated indices to guarantee uniform shape.
        :param ctx:
            Context manager.
        :param pcs: [batch_size, 3, num_points]
            Input point clouds.
        :param centroids: [batch_size, 3, num_centroids]
            Input centroids.
        :param radius: float
            Ball radius within which to sample ball points.
        :param max_samples: int
            Number to sample around each centroid.
        :return: group_idx: torch.LongTensor [batch_size, num_centroids, max_samples]
            Indices sampled from point clouds.
        """
        pc_length = pcs.pow(2).sum(dim=1, keepdim=True)
        c_length = centroids.pow(2).sum(dim=1).unsqueeze(-1)
        # (batch_size, num_centroids, num_points)
        dists = centroids.permute(0, 2, 1).bmm(pcs).mul(-2).add(pc_length).add(c_length)
        dists.clamp_(min=0).sqrt_()
        # Fill all indices with minimal-distant one.  This guarantees that the
        # output is always valid.
        group_idx = dists.argmin(dim=-1, keepdim=True).repeat(1, 1, max_samples)
        for i in range(dists.size(0)):
            for j in range(dists.size(1)):
                idx = dists[i, j].le(radius).nonzero().view(-1)
                # randomly sample
                if idx.numel() > max_samples:
                    selected = torch.randperm(idx.numel())[:max_samples]
                    group_idx[i, j] = idx[selected]
                else:
                    group_idx[i, j, :idx.numel()] = idx
        ctx.mark_non_differentiable(group_idx)
        return group_idx


if __name__ == "__main__":
    import time
    a = torch.randn(32, 3, 1024)
    a = a.cuda()
    tic = time.time()
    idx = FarthestPointSampleFunction.apply(a, 512)
    print('Elapsed time: %.3fs' % (time.time()-tic))
    from IPython import embed; embed()