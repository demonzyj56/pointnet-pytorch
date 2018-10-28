import itertools
import logging
import sys
import torch
from .configs import cfg

logger = logging.getLogger(__name__)


class PytorchSolverBase(object):

    def __init__(net, params=None):
        self.net = net
        self.params = params if params is not None else net.parameters()
        self.iterations = 0
        if cfg.OPTIMIZER == 'sgd':
            self.optimizer = torch.optim.SGD(self.params, lr=cfg.LR,
                                             momentum=cfg.MOMENTUM,
                                             weight_decay=cfg.WEIGHT_DECAY)
        elif cfg.OPTIMIZER == 'adam':
            self.optimizer = torch.optim.Adam(self.params, lr=cfg.LR,
                                              weight_decay=cfg.WEIGHT_DECAY)
        else:
            raise KeyError('Unknown optimizer: {}'.format(cfg.OPTIMIZER))
        if cfg.LR_POLICY == 'fixed':
            self.lr_steps = itertools.repeat(sys.maxsize)
        elif cfg.LR_POLICY == 'step':
            assert cfg.STEPSIZE > 0
            self.lr_steps = itertools.accumulate(itertools.repeat(cfg.STEPSIZE))
        elif cfg.LR_POLICY == 'multistep':
            assert len(cfg.STEPVALUE) > 0
            self.lr_steps = itertools.chain(cfg.STEPVALUE,
                                            itertools.repeat(sys.maxsize))
        elif cfg.LR_POLICY == 'exp':
            self.lr_steps = itertools.accumulate(itertools.repeat(1))
        else:
            raise KeyError('Unknown lr_policy: {}'.format(lr_policy))

    def _get_lr(self):
        raise NotImplementedError

    def _get_bn_momentum(self):
        raise NotImplementedError

    def adjust_lr(self):
        raise NotImplementedError

    def adjust_bn_momentum(self):
        raise NotImplementedError
