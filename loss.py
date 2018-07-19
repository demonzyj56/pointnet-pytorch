"""Loss function."""
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TransformRegLoss(nn.Module):
    """Regularization loss for transformation matrix."""

    def __init__(self, size_average=True):
        super(TransformRegLoss, self).__init__()
        self.size_average = size_average

    def forward(self, input):
        K = input.size(-1)
        it = input.clone().permute(0, 2, 1)
        out = it.bmm(input).view(it.size(0), -1)
        target = torch.eye(K).view(1, -1).expand_as(out).to(out.device)
        loss = out.sub(target).pow(2).sum(dim=1, keepdim=False)
        if self.size_average:
            loss = loss.mean()
        return loss

