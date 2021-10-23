from graphgym.config import cfg

import torch.optim as optim
import tensorflow.keras.optimizers as tf_optim
import tensorflow_addons as tfa
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import keras_export

from graphgym.contrib.optimizer import *
import graphgym.register as register


@keras_export("keras.optimizers.schedules.PiecewiseConstantDecay")
class StepLR(tf_optim.schedules.LearningRateSchedule):
    """
    TODO: Test this to see if it works. - JB
    """
    def __init__(
            self,
            initial_learning_rate,
            step_size,
            gamma=0.1,
            name=None):
        super(StepLR, self).__init__()

        self.initial_learning_rate = initial_learning_rate
        self.step_size = step_size
        self.gamma = gamma
        self.name = name

    def __call__(self, step):
        with ops.name_scope_v2(self.name or "StepLR"):
            initial_learning_rate = ops.convert_to_tensor_v2_with_dispatch(
                self.initial_learning_rate, name="initial_learning_rate")
            x_recomp = ops.convert_to_tensor_v2_with_dispatch(step)
            dtype = initial_learning_rate.dtype
            # TODO: possibly implement a floor function instead of '//' to ensure that we have the staircase style.
            return math_ops.multiply(initial_learning_rate, math_ops.pow(initial_learning_rate, step // self.step_size))

    def get_config(self):
        return {
            "boundaries": self.step_size,
            "values": self.gamma,
            "name": self.name
        }


def create_optimizer(params):
    """
    TODO: make sure that the information from "params" is accounted for - JB
    TODO: Test to make sure the learning rate and weight decay are working correctly. - JB
    """
    params = filter(lambda p: p.requires_grad, params)
    # Try to load customized optimizer
    for func in register.optimizer_dict.values():
        optimizer = func(params)
        if optimizer is not None:
            return optimizer
    if cfg.optim.optimizer == 'adam':
        if cfg.datasets.format == 'TfG':
            optimizer = tfa.optimizers.AdamW(learning_rate=cfg.optim.base_lr,
                                             weight_decay=cfg.optim.weight_decay)
        else:
            optimizer = optim.Adam(params, lr=cfg.optim.base_lr,
                                   weight_decay=cfg.optim.weight_decay)
    elif cfg.optim.optimizer == 'sgd':
        if cfg.datasets.format == 'TfG':
            optimizer = tfa.optimizers.SGDW(learning_rate=cfg.optim.base_lr,
                                            momentum=cfg.optim.momentum,
                                            weight_decay=cfg.optim.weight_decay)
        else:
            optimizer = optim.SGD(params, lr=cfg.optim.base_lr,
                                  momentum=cfg.optim.momentum,
                                  weight_decay=cfg.optim.weight_decay)
    else:
        raise ValueError('Optimizer {} not supported'.format(
            cfg.optim.optimizer))

    return optimizer


def create_scheduler(optimizer):
    """
    TODO: test to make sure that the scheduler gets applied to the optimizer correctly. - JB
    """
    # Try to load customized scheduler
    for func in register.scheduler_dict.values():
        scheduler = func(optimizer)
        if scheduler is not None:
            return scheduler
    if cfg.optim.scheduler == 'none':
        if cfg.datasets.format == 'TfG':
            scheduler = StepLR(initial_learning_rate=cfg.optim.base_lr,
                               step_size=cfg.optim.max_epoch + 1)
            optimizer.learning_rate = scheduler
        else:
            scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                  step_size=cfg.optim.max_epoch + 1)
    elif cfg.optim.scheduler == 'step':
        # This could use https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/PiecewiseConstantDecay
        # We will need to manually calculate the learning rate decay for each boundary region from the gamma value
        if cfg.datasets.format == 'TfG':
            scheduler = tf_optim.schedules.PiecewiseConstantDecay(boundaries=cfg.optim.steps,
                                          values=[
                                              cfg.optim.base_lr * cfg.optim.lr_decay ** i for i, _ in enumerate(cfg.optim.steps)
                                          ])  # Simulate MultiStepLR
            optimizer.learning_rate = scheduler
        else:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                       milestones=cfg.optim.steps,
                                                       gamma=cfg.optim.lr_decay)
    elif cfg.optim.scheduler == 'cos':
        if cfg.datasets.format == 'TfG':
            raise ValueError('Scheduler {} not supported for format {}'.format(
                cfg.optim.scheduler, cfg.datasets.format))
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                             T_max=cfg.optim.max_epoch)
    else:
        raise ValueError('Scheduler {} not supported'.format(
            cfg.optim.scheduler))
    return scheduler
