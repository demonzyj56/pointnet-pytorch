import itertools
import logging
import sys
import torch
from .configs import cfg

logger = logging.getLogger(__name__)


class PytorchSolverBase(object):

    def __init__(self, net, params=None):
        self.net = net
        self.params = params if params is not None else net.parameters()
        self.iterations = 0
        if cfg.OPTIMIZER == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.params, lr=cfg.LR, momentum=cfg.MOMENTUM,
                weight_decay=cfg.WEIGHT_DECAY
            )
        elif cfg.OPTIMIZER == 'adam':
            self.optimizer = torch.optim.Adam(
                self.params, lr=cfg.LR, betas=(cfg.MOMENTUM, cfg.MOMENTUM2),
                weight_decay=cfg.WEIGHT_DECAY
            )
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
            raise KeyError('Unknown lr_policy: {}'.format(cfg.LR_POLICY))
        self.iterations = 0
        self.next_step = next(self.lr_steps)
        self.bn_steps = cfg.BN.STEPSIZE if cfg.BN.STEPSIZE > 0 else sys.maxsize
        self.snapshot_iter = cfg.SNAPSHOT.ITER if cfg.SNAPSHOT.ITER > 0 else \
            sys.maxsize

    def _get_lr(self, cur_lr):
        """Get the learning rate. The iterations counter is assumed to be
        the current one, i.e. it has been increased."""
        if cfg.LR_POLICY == 'fixed':
            lr = cur_lr
        elif cfg.LR_POLICY == 'step' or cfg.LR_POLICY == 'multistep' or \
                cfg.LR_POLICY == 'exp':
            lr = cur_lr * cfg.GAMMA
        else:
            raise KeyError('Unknown lr_policy: {}'.format(cfg.LR_POLICY))
        return lr

    def adjust_lr(self):
        """Adjust the learning rate, AFTER increasing iteration counter."""
        if self.iterations == self.next_step:
            for param_group in self.optimizer.param_groups:
                new_lr = max(self._get_lr(param_group['lr']), cfg.MIN_LR)
                logger.info('Iteration: %d, update lr: %g -> %g',
                            self.iterations, param_group['lr'], new_lr)
                param_group['lr'] = new_lr
            self.next_step = next(self.lr_steps)

    def adjust_bn_momentum(self):
        """Adjust the batch norm momentum, AFTER increasing iteration number."""
        if self.iterations % self.bn_steps == 0:
            for m in self.net.modules():
                if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                                  torch.nn.BatchNorm3d)):
                    m.momentum = max(m.momentum*cfg.BN.GAMMA,
                                     cfg.BN.MIN_MOMENTUM)

    def snapshot(self):
        if self.iterations % self.snapshot_iter == 0:
            filename = cfg.SNAPSHOT.TEMPLATE.format(self.iterations)
            logger.info('Saving model to %s', filename)
            torch.save(self.net.state_dict(), filename)
