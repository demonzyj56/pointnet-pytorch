#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ModelNet training script."""
import argparse
import datetime
import logging
import pprint
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from pointnet2 import PointNet2SSG, PointNet2MSG
from modelnet.modelnet import ModelNetCls, PCAugmentation, collate_fn


def parse_args():
    """Argument parser."""
    parser = argparse.ArgumentParser(description='Argument parser for training ModelNet 40')
    parser.add_argument('--data_path', type=str, default=None, help='Path for modelnet data')
    parser.add_argument('--root_path', type=str, default='.default', help='Root path to save everything')
    parser.add_argument('--dataset', type=str, default='modelnet40', help='Which dataset to train and test on')
    parser.add_argument('--cuda_on', type=bool, default=True, help='Whether to train and test on GPUs')
    parser.add_argument('--rng_seed', type=int, default=-1, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--num_workers', type=int, default=4, help='No of workers for data loading')
    parser.add_argument('--epochs', type=int, default=250, help='Total epochs to go through for training')
    parser.add_argument('--lr', '--learning-rate', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='Minimal value of learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='lr momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for optimizer')
    parser.add_argument('--gamma', type=float, default=0.7, help='Gamma update for optimizer')
    parser.add_argument('--stepsize', type=int, default=20, help='How many epochs should decrease lr')
    parser.add_argument('--optimizer', type=str, default='adam', help='Which optimizer to use (SGD/ADAM)')
    parser.add_argument('--num_points', type=int, default=1024, help='No of datapoints for each model')
    parser.add_argument('--snapshot_interval', type=int, default=50, help='How many epochs should make a snapshot')
    parser.add_argument('--test_interval', type=int, default=1, help='How many epochs should run test; negative means dont run')
    parser.add_argument('--test_batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--multigpu', type=bool, default=False, help='Whether to train on multiple gpus')
    parser.add_argument('--bn_momentum', type=float, default=0.5, help='Initial value of bn momentum')
    parser.add_argument('--bn_stepsize', type=int, default=20, help='How many epoch should decrease bn momentum')
    parser.add_argument('--bn_gamma', type=float, default=0.5, help='Drease factor for bn momentum update')
    parser.add_argument('--bn_min_momentum', type=float, default=0.01, help='Minimal value of bn momentum')
    parser.add_argument('--use_msg', action='store_true', help='Use multi-scale grouping')
    args = parser.parse_args()
    return args


def setup_logging(name, filename=None):
    """Utility for every script to call on top-level.
    If filename is not None, then also log to the filename."""
    FORMAT = '[%(levelname)s %(asctime)s] %(filename)s:%(lineno)4d: %(message)s'
    DATEFMT = '%Y-%m-%d %H:%M:%S'
    logging.root.handlers = []
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    if filename is not None:
        handlers.append(logging.FileHandler(filename, mode='w'))
    logging.basicConfig(
        level=logging.INFO,
        format=FORMAT,
        datefmt=DATEFMT,
        handlers=handlers
    )
    logger = logging.getLogger(name)

    return logger


def train_model(args):
    """Main function for training classification model."""
    dataset = ModelNetCls(args.data_path, modelnet40=(args.dataset=='modelnet40'),
                          train=True, transform=PCAugmentation(),
                          num_points=args.num_points)
    if args.use_msg:
        point_net = PointNet2MSG(3, 40)
    else:
        point_net = PointNet2SSG(3, 40)
    if args.cuda_on:
        point_net = point_net.cuda()
    for m in point_net.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            m.momentum = args.bn_momentum
    if args.multigpu:
        point_net = torch.nn.DataParallel(point_net)
    if args.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(point_net.parameters(), lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(point_net.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
    else:
        raise ValueError('Unknown optimizer: {}'.format(args.optimizer))
    logger = logging.getLogger(__name__)
    for e in range(args.epochs):
        etic = time.time()
        point_net.train()
        logger.info('Training on epoch %d/%d', e+1, args.epochs)
        loader = torch.utils.data.DataLoader(
            dataset, args.batch_size, num_workers=args.num_workers,
            shuffle=True, collate_fn=collate_fn, pin_memory=True,
            drop_last=True,
        )
        tic = time.time()
        for batch_idx, (data, labels) in enumerate(loader):
            optimizer.zero_grad()
            if args.cuda_on:
                data, labels = data.cuda(), labels.cuda()
            out = point_net(data)
            _, predicted = out.max(dim=-1)
            loss = F.cross_entropy(out, labels)
            train_accuracy = (predicted==labels).sum().item() / len(labels)
            if time.time() - tic > 5:  # 5 sec
                logger.info(
                    '%d/%d for epoch %d, '
                    'Cls loss: %.3f, '
                    'train acc: %.3f',
                    batch_idx*args.batch_size, len(dataset), e+1,
                    loss.item(), train_accuracy
                )
                tic = time.time()
            loss.backward()
            optimizer.step()
        if args.snapshot_interval > 0 and \
                ((e + 1) % args.snapshot_interval == 0):
            filename = os.path.join(args.root_path, '{}.{}-{}.pth'.format(
                'PointNet2Cls', args.dataset, e+1
            ))
            logger.info('Saving model to %s', filename)
            torch.save(point_net.state_dict(), filename)
        if args.test_interval > 0 and ((e+1) % args.test_interval == 0):
            logger.info('Running test for epoch %d/%d', e+1, args.epochs)
            ins_acc, cls_acc = test_model(point_net, args)
            logger.info('Instance accuracy: %.3f, class accuracy: %.3f',
                        ins_acc, cls_acc)
        # update learning rate
        if (args.stepsize > 0) and ((e+1) % args.stepsize == 0):
            args.lr = max(args.lr * args.gamma, args.min_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
            logger.info('Learning rate set to %g', args.lr)
        # update bn momentum
        if (args.bn_stepsize > 0) and ((e+1) % args.bn_stepsize == 0):
            args.bn_momentum = max(args.bn_momentum*args.bn_gamma,
                                   args.bn_min_momentum)
            for m in point_net.modules():
                if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    m.momentum = args.bn_momentum
            logger.info('BatchNorm momentum set to %g', args.bn_momentum)
        logger.info('Elapsed time for epoch %d: %.3fs', e+1, time.time()-etic)
    if args.snapshot_interval > 0:
        filename = os.path.join(args.root_path, '{}.{}.pth'.format(
            'PointNet2Cls', args.dataset
        ))
        logger.info('Saving final model to %s', filename)
        torch.save(point_net.state_dict(), filename)


def test_model(model, args):
    """Run test on model."""
    model.eval()
    dataset = ModelNetCls(args.data_path, modelnet40=(args.dataset=='modelnet40'),
                          train=False, transform=None,
                          num_points=args.num_points)
    loader = torch.utils.data.DataLoader(
        dataset, args.test_batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_fn, pin_memory=True, drop_last=False
    )
    num_classes = len(np.unique(dataset.labels))
    num_per_class = [(dataset.labels==i).sum() for i in range(num_classes)]
    class_hit = [0 for _ in range(num_classes)]
    for data, labels in loader:
        if args.cuda_on:
            data = data.cuda()
        out = model(data)
        _, predicted = out.max(dim=-1)
        predicted = predicted.tolist()
        labels = labels.tolist()
        for p, t in zip(predicted, labels):
            if p == t:
                class_hit[p] = class_hit[p] + 1
    instance_accuracy = sum(class_hit) / len(dataset)
    class_accuracies = [n/total for n, total in zip(class_hit, num_per_class)]
    return instance_accuracy, np.mean(class_accuracies)


def main():
    args = parse_args()
    if not os.path.exists(args.root_path):
        os.makedirs(args.root_path)
    log_name = os.path.join(
        args.root_path,
        '{:s}.{:%Y-%m-%d_%H-%M-%S}.{:s}.log'.format(
            args.dataset,
            datetime.datetime.now(),
            'train_test' if args.test_interval > 0 else 'train'
        )
    )
    logger = setup_logging(__name__, log_name)
    logger.info(pprint.pformat(args))
    if args.rng_seed >= 0:
        np.random.seed(args.rng_seed)
        torch.manual_seed(args.rng_seed)
        torch.cuda.manual_seed_all(args.rng_seed)
    train_model(args)


if __name__ == "__main__":
    main()
