import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
import tensorflow as tf
import tf_geometric as tfg

from graphgym.config import cfg
from graphgym.models.act import act_dict
from graphgym.contrib.layer.generalconv import (GeneralConvLayer,
                                                GeneralEdgeConvLayer)

from graphgym.contrib.layer import *
import graphgym.register as register


## General classes
class GeneralLayer(nn.Module):
    """
    TODO: convert all of these classes to work with tensorflow. - JB
    """
    '''General wrapper for layers'''

    def __init__(self, name, dim_in, dim_out, has_act=True, has_bn=True,
                 has_l2norm=False, **kwargs):
        super(GeneralLayer, self).__init__()
        self.has_l2norm = has_l2norm
        has_bn = has_bn and cfg.gnn.batchnorm
        self.layer = layer_dict[name](dim_in, dim_out,
                                      bias=not has_bn, **kwargs)
        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(nn.BatchNorm1d(
                dim_out, eps=cfg.bn.eps, momentum=cfg.bn.mom))
        if cfg.gnn.dropout > 0:
            layer_wrapper.append(nn.Dropout(
                p=cfg.gnn.dropout, inplace=cfg.mem.inplace))
        if has_act:
            layer_wrapper.append(act_dict[cfg.gnn.act])
        self.post_layer = nn.Sequential(*layer_wrapper)

    def forward(self, batch):
        batch = self.layer(batch)
        if isinstance(batch, torch.Tensor):
            batch = self.post_layer(batch)
            if self.has_l2norm:
                batch = F.normalize(batch, p=2, dim=1)
        else:
            batch.node_feature = self.post_layer(batch.node_feature)
            if self.has_l2norm:
                batch.node_feature = F.normalize(batch.node_feature, p=2, dim=1)
        return batch


class TFGeneralLayer(tf.keras.layers.Layer):
    def __init__(self, name, dim_in, dim_out, has_act=True, has_bn=True,
                 has_l2norm=False, **kwargs):
        super(TFGeneralLayer, self).__init__()
        self.has_l2norm = has_l2norm
        has_bn = has_bn and cfg.gnn.batchnorm
        self.sequential_layers = list()
        self.sequential_layers.append(layer_dict[name](dim_in, dim_out,
                                                       bias=not has_bn, **kwargs))

        if self.has_l2norm:
            self.sequential_layers.append(tf.keras.layers.Normalization(axis=1, p=2))
        if has_bn:
            # Axis may need to be '1'
            self.sequential_layers.append(tf.keras.layers.BatchNormalization(axis=0,
                                                                             epsilon=cfg.bn.eps,
                                                                             momentum=cfg.bn.mom))
        if cfg.gnn.dropout > 0:
            self.sequential_layers.append(tf.keras.layers.Dropoout(cfg.gnn.dropout))
        if has_act:
            self.sequential_layers.append(act_dict[cfg.gnn.act])

    def call(self, inputs):
        h = inputs
        for layer in self.sequential_layers:
            h = layer(inputs)
        if self.has_l2norm:
            h = tf.math.l2_normalize(h)  # TODO: Confirm that this is correct - JB
        return h


class GeneralMultiLayer(nn.Module):
    """
    TODO: Convert this to tensorflow - JB
    """
    '''General wrapper for stack of layers'''

    def __init__(self, name, num_layers, dim_in, dim_out, dim_inner=None,
                 final_act=True, **kwargs):
        super(GeneralMultiLayer, self).__init__()
        dim_inner = dim_in if dim_inner is None else dim_inner
        for i in range(num_layers):
            d_in = dim_in if i == 0 else dim_inner
            d_out = dim_out if i == num_layers - 1 else dim_inner
            has_act = final_act if i == num_layers - 1 else True
            layer = GeneralLayer(name, d_in, d_out, has_act, **kwargs)
            self.add_module('Layer_{}'.format(i), layer)

    def forward(self, batch):
        for layer in self.children():
            batch = layer(batch)
        return batch


class TFGeneralMultiLayer(tf.keras.layers.Layer):
    def __init__(self, name, num_layers, dim_in, dim_out, dim_inner=None,
                 final_act=True, **kwargs):
        super(TFGeneralMultiLayer, self).__init__()
        self.sequential_layers = list()
        dim_inner = dim_in if dim_inner is None else dim_inner
        d_in = dim_in
        for i in range(num_layers):
            d_out = dim_out if i == num_layers - 1 else dim_inner
            has_act = final_act if i == num_layers - 1 else True
            self.sequential_layers.append(TFGeneralLayer(name, d_in, d_out, has_act, **kwargs))

    def call(self, inputs):
        h = inputs
        for layer in self.sequential_layers:
            h = layer(h)
        return h


## Core basic layers
# Input: batch; Output: batch
class Linear(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(Linear, self).__init__()
        self.model = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = self.model(batch)
        else:
            batch.node_feature = self.model(batch.node_feature)
        return batch


class TFLinear(tf.keras.layers.Layer):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(TFLinear, self).__init__()
        self.internal_layer = tf.keras.layers.Dense(dim_out, bias=bias)

    def call(self, inputs):
        return self.internal_layer(inputs)


class BatchNorm1dNode(nn.Module):
    '''General wrapper for layers'''

    def __init__(self, dim_in):
        super(BatchNorm1dNode, self).__init__()
        self.bn = nn.BatchNorm1d(dim_in, eps=cfg.bn.eps, momentum=cfg.bn.mom)

    def forward(self, batch):
        batch.node_feature = self.bn(batch.node_feature)
        return batch


class BatchNorm1dEdge(nn.Module):
    '''General wrapper for layers'''

    def __init__(self, dim_in):
        super(BatchNorm1dEdge, self).__init__()
        self.bn = nn.BatchNorm1d(dim_in, eps=cfg.bn.eps, momentum=cfg.bn.mom)

    def forward(self, batch):
        batch.edge_feature = self.bn(batch.edge_feature)
        return batch


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True, dim_inner=None,
                 num_layers=2, **kwargs):
        '''
        Note: MLP works for 0 layers
        '''
        super(MLP, self).__init__()
        dim_inner = dim_in if dim_inner is None else dim_inner
        layers = []
        if num_layers > 1:
            layers.append(
                GeneralMultiLayer('linear', num_layers - 1, dim_in, dim_inner,
                                  dim_inner, final_act=True))
            layers.append(Linear(dim_inner, dim_out, bias))
        else:
            layers.append(Linear(dim_in, dim_out, bias))
        self.model = nn.Sequential(*layers)

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = self.model(batch)
        else:
            batch.node_feature = self.model(batch.node_feature)
        return batch


class TFMLP(tf.keras.layers.Layer):
    def __init__(self, dim_in, dim_out, bias=True, dim_inner=None,
                 num_layers=2, **kwargs):
        '''
        Note: MLP works for 0 layers
        '''
        super(TFMLP, self).__init__()
        dim_inner = dim_in if dim_inner is None else dim_inner
        self.sequential_layers = list()
        if num_layers > 1:
            self.sequential_layers.append(TFGeneralMultiLayer('linear', num_layers - 1, dim_in, dim_inner,
                                                              dim_inner, final_act=True))
            self.sequential_layers.append(TFLinear(dim_inner, dim_out, bias))
        else:
            self.sequential_layers.append(TFLinear(dim_in, dim_out, bias))

    def call(self, inputs):
        h = inputs
        for layer in self.sequential_layers:
            h = layer(h)
        return h


class GCNConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GCNConv, self).__init__()
        self.model = pyg.nn.GCNConv(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index)
        return batch


class TFGCNConv(tf.keras.layers.Layer):
    pass


class SAGEConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(SAGEConv, self).__init__()
        self.model = pyg.nn.SAGEConv(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index)
        return batch


class TFSAGEConv(tf.keras.layers.Layer):
    pass


class GATConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GATConv, self).__init__()
        self.model = pyg.nn.GATConv(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index)
        return batch


class TFGATConv(tf.keras.layers.Layer):
    pass


class GINConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GINConv, self).__init__()
        gin_nn = nn.Sequential(nn.Linear(dim_in, dim_out), nn.ReLU(),
                               nn.Linear(dim_out, dim_out))
        self.model = pyg.nn.GINConv(gin_nn)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index)
        return batch


class TFGINConv(tf.keras.layers.Layer):
    pass


class SplineConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(SplineConv, self).__init__()
        self.model = pyg.nn.SplineConv(dim_in, dim_out,
                                       dim=1, kernel_size=2, bias=bias)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index,
                                        batch.edge_feature)
        return batch


class TFSplineConv(tf.keras.layers.Layer):
    pass


class GeneralConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GeneralConv, self).__init__()
        self.model = GeneralConvLayer(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index)
        return batch


class TFGeneralConv(tf.keras.layers.Layer):
    pass


class GeneralEdgeConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GeneralEdgeConv, self).__init__()
        self.model = GeneralEdgeConvLayer(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index,
                                        edge_feature=batch.edge_feature)
        return batch


class TFGeneralEdgeConv(tf.keras.layers.Layer):
    pass


class GeneralSampleEdgeConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GeneralSampleEdgeConv, self).__init__()
        self.model = GeneralEdgeConvLayer(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        edge_mask = torch.rand(batch.edge_index.shape[1]) < cfg.gnn.keep_edge
        edge_index = batch.edge_index[:, edge_mask]
        edge_feature = batch.edge_feature[edge_mask, :]
        batch.node_feature = self.model(batch.node_feature, edge_index,
                                        edge_feature=edge_feature)
        return batch


class TFGeneralSampleEdgeConv(tf.keras.layers.Layer):
    pass


layer_dict = {
    'linear': TFLinear if cfg.dataset.format == 'TfG' else Linear,
    'mlp': TFMLP if cfg.dataset.format == 'TfG' else MLP,
    'gcnconv': TFGCNConv if cfg.dataset.format == 'TfG' else  GCNConv,
    'sageconv': TFSAGEConv if cfg.dataset.format == 'TfG' else SAGEConv,
    'gatconv': TFGATConv if cfg.dataset.format == 'TfG' else GATConv,
    'splineconv': TFSplineConv if cfg.dataset.format == 'TfG' else SplineConv,
    'ginconv': TFGINConv if cfg.dataset.format == 'TfG' else GINConv,
    'generalconv': TFGeneralConv if cfg.dataset.format == 'TfG' else GeneralConv,
    'generaledgeconv': TFGeneralEdgeConv if cfg.dataset.format == 'TfG' else GeneralEdgeConv,
    'generalsampleedgeconv': TFGeneralSampleEdgeConv if cfg.dataset.format == 'TfG' else GeneralSampleEdgeConv,
}

# register additional convs
layer_dict = {**register.layer_dict, **layer_dict}
