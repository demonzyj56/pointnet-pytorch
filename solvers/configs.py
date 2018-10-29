"""Solver configs."""
import logging
import os
from chino.frozen_dict import FrozenDict

logger = logging.getLogger(__name__)
__C = FrozenDict()
cfg = __C  # for importing

# The name of the solver
__C.NAME = 'default'

# Learning rate decay policy, available options are:
#   - fixed
#   - step
#   - multistep
#   - exp
__C.LR_POLICY = 'step'

# learning rate
__C.LR = 1e-2

# optional min lr for step-wise decay
__C.MIN_LR = -1.

# momentum
__C.MOMENTUM = 0.9

# weight decay
__C.WEIGHT_DECAY = 5e-4

# gamma for decay on each step
__C.GAMMA = 0.9

# how many iterations should the learning be adjusted
__C.STEPSIZE = -1

# step value controls the size of each step
__C.STEPVALUE = []

# optimizer type.  Available types are:
#   - sgd
#   - adam
__C.OPTIMIZER = 'sgd'

# second momentum value for adam
__C.MOMENTUM2 = 0.999

# batch normalization momentum
__C.BN.MOMENTUM = 0.01

# optional min momentum for step-wise decay
__C.BN.MIN_MOMENTUM = -1.

# only step-wise decay of batchnorm momentum is also implemented
__C.BN.STEPSIZE = -1

# decay rate for momentum
__C.BN.GAMMA = 0.9


# For how many iterations should snapshot the network.
__C.SNAPSHOT.ITER = -1

# Snapshot template
__C.SNAPSHOT.TEMPLATE = os.path.join(__C.OUTPUT_PATH,
                                     __C.NAME + "-{}.pth")


#####################################
# Freezing the configs.
#####################################
__C.freeze()
