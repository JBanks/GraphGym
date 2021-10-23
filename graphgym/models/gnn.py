import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import tf_geometric as tfg
import tf_geometric.nn as tfg_nn

from graphgym.config import cfg
from graphgym.models.head import head_dict
from graphgym.models.layer import (GeneralLayer, TFGeneralLayer, GeneralMultiLayer, TFGeneralMultiLayer,
                                   BatchNorm1dNode, BatchNorm1dEdge)
from graphgym.models.act import act_dict
from graphgym.models.feature_augment import Preprocess
from graphgym.init import init_weights
from graphgym.models.feature_encoder import node_encoder_dict, edge_encoder_dict

from graphgym.contrib.stage import *
import graphgym.register as register


########### Layer ############
def GNNLayer(dim_in, dim_out, has_act=True):
    if cfg.dataset.format == 'TfG':
        return TFGeneralLayer(cfg.gnn.layer_type, dim_in, dim_out, has_act)
    else:
        return GeneralLayer(cfg.gnn.layer_type, dim_in, dim_out, has_act)


def GNNPreMP(dim_in, dim_out):
    if cfg.dataset.format == 'TfG':
        return None
    else:
        return GeneralMultiLayer('linear', cfg.gnn.layers_pre_mp,
                                 dim_in, dim_out, dim_inner=dim_out, final_act=True)


########### Block: multiple layers ############

class GNNSkipBlock(nn.Module):
    '''Skip block for GNN'''

    def __init__(self, dim_in, dim_out, num_layers):
        super(GNNSkipBlock, self).__init__()
        if num_layers == 1:
            self.f = [GNNLayer(dim_in, dim_out, has_act=False)]
        else:
            self.f = []
            for i in range(num_layers - 1):
                d_in = dim_in if i == 0 else dim_out
                self.f.append(GNNLayer(d_in, dim_out))
            d_in = dim_in if num_layers == 1 else dim_out
            self.f.append(GNNLayer(d_in, dim_out, has_act=False))
        self.f = nn.Sequential(*self.f)
        self.act = act_dict[cfg.gnn.act]
        if cfg.gnn.stage_type == 'skipsum':
            assert dim_in == dim_out, 'Sum skip must have same dim_in, dim_out'

    def forward(self, batch):
        node_feature = batch.node_feature
        if cfg.gnn.stage_type == 'skipsum':
            batch.node_feature = \
                node_feature + self.f(batch).node_feature
        elif cfg.gnn.stage_type == 'skipconcat':
            batch.node_feature = \
                torch.cat((node_feature, self.f(batch).node_feature), 1)
        else:
            raise ValueError('cfg.gnn.stage_type must in [skipsum, skipconcat]')
        batch.node_feature = self.act(batch.node_feature)
        return batch


########### Stage: NN except start and head ############

class GNNStackStage(nn.Module):
    '''Simple Stage that stack GNN layers'''

    def __init__(self, dim_in, dim_out, num_layers):
        super(GNNStackStage, self).__init__()
        for i in range(num_layers):
            d_in = dim_in if i == 0 else dim_out
            layer = GNNLayer(d_in, dim_out)
            self.add_module('layer{}'.format(i), layer)
        self.dim_out = dim_out

    def forward(self, batch):
        for layer in self.children():
            batch = layer(batch)
        if cfg.gnn.l2norm:
            batch.node_feature = F.normalize(batch.node_feature, p=2, dim=-1)
        return batch


class TFGNNStackStage(tf.keras.layers.Layer):
    """
    TODO: Implement this in tensorflow (high priority) - JB
    Perhaps this should be a wrapper to build one layer out of many layers
    """
    def __init__(self, dim_in, dim_out, num_layers):
        super(TFGNNStackStage, self).__init()
        d_in = dim_in
        self.stack_list = list()
        for i in range(num_layers):
            self.stack_list.append(GNNLayer(d_in, dim_out))
            d_in = dim_out

    def call(self, inputs):
        h = inputs
        for layer in self.stack_list:
            h = layer(h)
        if cfg.gnn.l2norm:
            pass  # TODO: normalize for l2norm
        return h


class GNNSkipStage(nn.Module):
    ''' Stage with skip connections'''

    def __init__(self, dim_in, dim_out, num_layers):
        super(GNNSkipStage, self).__init__()
        assert num_layers % cfg.gnn.skip_every == 0, \
            'cfg.gnn.skip_every must be multiples of cfg.gnn.layer_mp' \
            '(excluding head layer)'
        for i in range(num_layers // cfg.gnn.skip_every):
            if cfg.gnn.stage_type == 'skipsum':
                d_in = dim_in if i == 0 else dim_out
            elif cfg.gnn.stage_type == 'skipconcat':
                d_in = dim_in if i == 0 else dim_in + i * dim_out
            block = GNNSkipBlock(d_in, dim_out, cfg.gnn.skip_every)
            self.add_module('block{}'.format(i), block)
        if cfg.gnn.stage_type == 'skipconcat':
            self.dim_out = d_in + dim_out
        else:
            self.dim_out = dim_out

    def forward(self, batch):
        for layer in self.children():
            batch = layer(batch)
        if cfg.gnn.l2norm:
            batch.node_feature = F.normalize(batch.node_feature, p=2, dim=-1)
        return batch


stage_dict = {
    'stack': TFGNNStackStage if cfg.dataset.format == 'TfG' else GNNStackStage,
    'skipsum': GNNSkipStage,
    'skipconcat': GNNSkipStage,
}

stage_dict = {**register.stage_dict, **stage_dict}


########### Model: start + stage + head ############

class GNN(nn.Module):
    '''General GNN model'''

    def __init__(self, dim_in, dim_out, **kwargs):
        """
            Parameters:
            node_encoding_classes - For integer features, gives the number
            of possible integer features to map.
        """
        super(GNN, self).__init__()
        GNNStage = stage_dict[cfg.gnn.stage_type]
        GNNHead = head_dict[cfg.dataset.task]

        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = node_encoder_dict[cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.dataset.encoder_dim)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(cfg.dataset.encoder_dim)
            # Update dim_in to reflect the new dimension fo the node features
            dim_in = cfg.dataset.encoder_dim
        if cfg.dataset.edge_encoder:
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = edge_encoder_dict[cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.dataset.encoder_dim)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dEdge(cfg.dataset.edge_dim)

        self.preprocess = Preprocess(dim_in)
        d_in = self.preprocess.dim_out
        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(d_in, cfg.gnn.dim_inner)
            d_in = cfg.gnn.dim_inner
        if cfg.gnn.layers_mp > 0:
            self.mp = GNNStage(dim_in=d_in,
                               dim_out=cfg.gnn.dim_inner,
                               num_layers=cfg.gnn.layers_mp)
            d_in = self.mp.dim_out
        self.post_mp = GNNHead(dim_in=d_in, dim_out=dim_out)

        self.apply(init_weights)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


class TFGNN(tf.keras.Model):
    """
    TODO: Complete this implementation.  Should use heads and stacks (maybe) - JB
    """
    def __init__(self, dim_in, dim_out, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # This section doesn't need to be split into its own function right now.
        d_in = dim_in
        self.stack_list = list()

        # Node and Edge encodings will be evaluated later if required.  by default they are both False

        # Pre-MessagePassing
        dim_inner = dim_in if cfg.gnn.dim_inner is None else cfg.gnn.dim_inner
        for i in range(cfg.gnn.layers_pre_mp):
            d_in = dim_in if i == 0 else dim_inner
            d_out = dim_out if i == cfg.gnn.layers_pre_mp - 1 else dim_inner
            has_act = True if i == cfg.gnn.layers_pre_mp - 1 else True
            self.stack_list.append(tf.keras.layers.Linear(d_out, use_bias=not cfg.gnn.batchnorm))
            if cfg.gnn.batchnorm:
                self.stack_list.append(tf.keras.layers.Batchnorm(dim_out))  # Add eps and momentum
            if cfg.gnn.dropout > 0:
                self.stack_list.append(tf.keras.layers.Dropoout(cfg.gnn.dropout))
            self.stack_list.append(act_dict[cfg.gnn.act])
            d_in = dim_out

        # MessagePassing
        for i in range(cfg.gnn.layers_mp):
            self.stack_list.append(tfg.layers.GCN(dim_out, use_bias=not cfg.gnn.batchnorm))
            # layer_dict[name](d_in, dim_out,
            #                           bias=not has_bn, **kwargs)
            if cfg.gnn.batchnorm:
                self.stack_list.append(tf.keras.layers.Batchnorm(dim_out))  # Add eps and momentum
            if cfg.gnn.dropout > 0:
                self.stack_list.append(tf.keras.layers.Dropoout(cfg.gnn.dropout))
            self.stack_list.append(act_dict[cfg.gnn.act])
            d_in = dim_out

        # Post-MessagePassing
        dim_inner = dim_in if cfg.gnn.dim_inner is None else cfg.gnn.dim_inner

        self.stack_list.append(tf.keras.layers.Linear(dim_inner,
                                                      num_layers=cfg.gnn.layers_post_mp, bias=True))

    def call(self, inputs, training=None, mask=None, cache=None):
        h, edge_index, edge_weight = inputs
        for layer in self.stack_list:
            h = self.dropout(h, training=training)
            h = layer([h, edge_index, edge_weight], cache=cache)

        return h
