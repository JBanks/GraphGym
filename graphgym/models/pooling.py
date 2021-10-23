import torch
from torch_scatter import scatter
import tensorflow as tf
import tf_geometric as tfg
from graphgym.config import cfg

from graphgym.contrib.pooling import *
import graphgym.register as register


# Pooling options (pool nodes into graph representations)
# pooling function takes in node embedding [num_nodes x emb_dim] and
# batch (indices) and outputs graph embedding [num_graphs x emb_dim].
def global_add_pool(x, batch, id=None, size=None):
    size = batch.max().item() + 1 if size is None else size
    if cfg.dataset.transform == 'ego':
        x = torch.index_select(x, dim=0, index=id)
        batch = torch.index_select(batch, dim=0, index=id)
    return scatter(x, batch, dim=0, dim_size=size, reduce='add')


def global_mean_pool(x, batch, id=None, size=None):
    size = batch.max().item() + 1 if size is None else size
    if cfg.dataset.transform == 'ego':
        x = torch.index_select(x, dim=0, index=id)
        batch = torch.index_select(batch, dim=0, index=id)
    return scatter(x, batch, dim=0, dim_size=size, reduce='mean')


def global_max_pool(x, batch, id=None, size=None):
    size = batch.max().item() + 1 if size is None else size
    if cfg.dataset.transform == 'ego':
        x = torch.index_select(x, dim=0, index=id)
        batch = torch.index_select(batch, dim=0, index=id)
    return scatter(x, batch, dim=0, dim_size=size, reduce='max')


def tf_global_add_pool(x, inputs, id=None, size=None):
    """
    TODO: make sure this works - JB
    Skip the egocentric piece until later.  Get the GNN up first.
    Maybe look into using the tfg.nn.pool.common_pool.sum or tf.math.unsorted_segment_sum
    """
    return tfg.layers.SumPool(inputs, training=None, mask=None)


def tf_global_mean_pool(x, inputs, id=None, size=None):
    """
    TODO: make sure this works - JB
    """
    return tfg.layers.MeanPool(inputs)


def tf_global_max_pool(x, inputs, id=None, size=None):
    """
    TODO: make sure this works - JB
    """
    return tfg.layers.MaxPool(inputs)


pooling_dict = {
    'add': tf_global_add_pool if cfg.dataset.format == 'TfG' else global_add_pool,
    'mean': tf_global_mean_pool if cfg.dataset.format == 'TfG' else global_mean_pool,
    'max': tf_global_max_pool if cfg.dataset.format == 'TfG' else global_max_pool
}

pooling_dict = {**register.pooling_dict, **pooling_dict}
