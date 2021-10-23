import torch.nn as nn
import tensorflow.keras.activations as tf_act
import tensorflow as tf
from functools import partial
from graphgym.config import cfg
from graphgym.contrib.act import *
import graphgym.register as register

TfG = cfg.dataset.format == 'TfG'


def tf_act_prelu():
    """
    TODO: implement this (low priority) - JB
    """
    pass


act_dict = {
    'relu': tf_act.relu if TfG else nn.ReLU(inplace=cfg.mem.inplace),
    'selu': tf_act.selu if TfG else nn.SELU(inplace=cfg.mem.inplace),
    'prelu': tf_act_prelu if TfG else nn.PReLU(),
    'elu': tf_act.elu if TfG else nn.ELU(inplace=cfg.mem.inplace),
    'lrelu_01': partial(tf.nn.leaky_relu, alpha=0.1) if TfG else
    nn.LeakyReLU(negative_slope=0.1, inplace=cfg.mem.inplace),  # partial with slope
    'lrelu_025': partial(tf.nn.leaky_relu, alpha=0.25) if TfG else
    nn.LeakyReLU(negative_slope=0.25, inplace=cfg.mem.inplace),
    'lrelu_05': partial(tf.nn.leaky_relu, alpha=0.5) if TfG else
    nn.LeakyReLU(negative_slope=0.5, inplace=cfg.mem.inplace),
}

act_dict = {**register.act_dict, **act_dict}
