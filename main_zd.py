import os
import random
import numpy as np
import torch
import logging

#from GraphGym.graphgym.cmd_args import parse_args
from graphgym.config import (cfg, assert_cfg, dump_cfg,
                  update_out_dir, get_parent_dir)
from graphgym.loader import create_dataset, create_loader
from graphgym.logger import setup_printing, create_logger
from graphgym.optimizer import create_optimizer, create_scheduler
from graphgym.model_builder import create_model
from graphgym.train import train
from graphgym.utils.agg_runs import agg_runs
from graphgym.utils.comp_budget import params_count
from graphgym.utils.device import auto_select_device
from graphgym.register import train_dict

import tensorflow as tf
import tf_geometric as tfg

repeat = 3


class GCNModel(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gcn0 = tfg.layers.GCN(128, activation=tf.nn.relu)
        self.gcn1 = tfg.layers.GCN(128, activation=tf.nn.relu)
        self.gcn2 = tfg.layers.GCN(128, activation=tf.nn.relu)
        self.mlp = tf.keras.models.Sequential([tf.keras.layers.Flatten()
                                               ,tf.keras.layers.Dense(256, activation='relu')
                                               ,tf.keras.layers.Dense(datasets[0].num_labels)])

    def call(self, inputs, training=None, mask=None, cache=None):
        x, edge_index, edge_weight = inputs
        h = self.gcn0([x, edge_index, edge_weight], cache=cache)  
        h = self.gcn1([h, edge_index, edge_weight], cache=cache)
        h = self.gcn2([h, edge_index, edge_weight], cache=cache)
        h = self.mlp(h)
        return h
    
    
class GATModel(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gat0 = tfg.layers.GAT(128, activation=tf.nn.relu)
        self.gat1 = tfg.layers.GAT(128, activation=tf.nn.relu)
        self.gat2 = tfg.layers.GAT(128, activation=tf.nn.relu)
        self.mlp = tf.keras.models.Sequential([tf.keras.layers.Flatten()
                                               ,tf.keras.layers.Dense(256, activation='relu')
                                               ,tf.keras.layers.Dense(datasets[0].num_labels)])

    def call(self, inputs, training=None, mask=None, cache=None):
        x, edge_index,_ = inputs
        h = self.gat0([x, edge_index], training=training)
        h = self.gat1([h, edge_index], training=training)
        h = self.gat2([h, edge_index], training=training)
        h = self.mlp(h)
        return h


class SAGEModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sage0 = tfg.layers.GCNGraphSage(128, activation=tf.nn.relu)
        self.sage1 = tfg.layers.GCNGraphSage(128, activation=tf.nn.relu)
        self.sage2 = tfg.layers.GCNGraphSage(128, activation=tf.nn.relu)
        self.mlp = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                               tf.keras.layers.Dense(256, activation='relu'),
                                               tf.keras.layers.Dense(datasets[0].num_labels)])

    def call(self, inputs, training=None, mask=None, cache=None):
        x, edge_index, _ = inputs
        h = self.sage0([x, edge_index], training=training)
        h = self.sage1([h, edge_index], training=training)
        h = self.sage2([h, edge_index], training=training)
        h = self.mlp(h)
        return h


class GINModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gin0 = tfg.layers.GIN(128, activation=tf.nn.relu)
        self.gin1 = tfg.layers.GIN(128, activation=tf.nn.relu)
        self.gin2 = tfg.layers.GIN(128, activation=tf.nn.relu)
        self.mlp = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                               tf.keras.layers.Dense(256, activation='relu'),
                                               tf.keras.layers.Dense(datasets[0].num_labels)])

    def call(self, inputs, training=None, mask=None, cache=None):
        x, edge_index, _ = inputs
        h = self.gin0([x, edge_index], training=training)
        h = self.gin1([h, edge_index], training=training)
        h = self.gin2([h, edge_index], training=training)
        h = self.mlp(h)
        return h


class APPNPModel(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.appnp = tfg.layers.APPNP([64, datasets[0].num_labels], alpha=0.1, num_iterations=10,
                                      dense_drop_rate=0, edge_drop_rate=0)
        #self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, inputs, training=None, mask=None, cache=None):
        x, edge_index, edge_weight = inputs
        #h = self.dropout(x, training=training)
        h = self.appnp([x, edge_index, edge_weight], training=training, cache=cache)
        return h

    
acc_lists = []
max_acc = []
for i in range(repeat):
    # Load config file
    cfg.merge_from_file('node.yaml')
    #cfg.device = 'cuda'
    #print(cfg.dataset.format)
    #cfg.merge_from_list(args.opts)
    assert_cfg(cfg)
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    out_dir_parent = cfg.out_dir
    cfg.seed = i + 1
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    #update_out_dir(out_dir_parent, args.cfg_file)
    dump_cfg(cfg)
    setup_printing()
    auto_select_device()
    # Set learning environment
    datasets = create_dataset()
    loaders = create_loader(datasets)
    meters = create_logger(datasets)
    #model = create_model(datasets)
    model_func = {
        'gcnidconv': GCNModel,
        'sageidconv': GCNModel,
        'gatidconv': GCNModel,
        'ginidconv': GCNModel,
        'gcnconv': GCNModel,
        'gatconv': GATModel,
        'sageconv': SAGEModel,
        'ginconv': GINModel,
    }
    model = model_func[cfg.gnn.layer_type](datasets)
    #optimizer = create_optimizer(model.parameters())
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.optim.base_lr)
    #print(cfg.optim.base_lr)
    #scheduler = create_scheduler(optimizer)
    #logging.info(model)
    #logging.info(cfg)
    cfg.params = 0
    
    #cfg.device = 'cuda'
    #print(cfg.dataset.augment_label_dims)
    #logging.info('Num parameters: {}'.format(cfg.params))    
    
    if cfg.train.mode == 'standard':
        acc_list = train(meters, loaders, model, optimizer,datasets)
        acc_lists.append(acc_list)
        max_acc.append(max(acc_list))
    else:
        train_dict[cfg.train.mode](
            meters, loaders, model, optimizer, scheduler)
np.savetxt('./' + cfg.out_dir+'/val'+'/middle'+f'/{cfg.model.type}{"Fast" if cfg.dataset.augment_label_dims > 0 else ""}-{cfg.gnn.layer_type}'+f'_{cfg.dataset.name}.txt', np.array(acc_lists))
np.savetxt('./' + cfg.out_dir+'/val'+'/final'+f'/{cfg.model.type}{"Fast" if cfg.dataset.augment_label_dims > 0 else ""}-{cfg.gnn.layer_type}'+f'_{cfg.dataset.name}_avg_acc.txt', np.array([np.mean(max_acc)]))
print(f'The average validation accuracy of {repeat} rounds is: {np.mean(max_acc)}')



