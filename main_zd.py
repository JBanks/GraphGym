import os
import random
import numpy as np
import torch
import logging
import argparse

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
from TfgIDLayer import IDGCN, IDGAT, IDSAGE, IDGIN  # These are the custom "ID-GNN" layers

repeat = 3  # the number of times to run each experiment.


class GCNModel(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Build out n GCN layers.  In the paper, they used 3, so we simply hard coded that many layers.
        self.gcn0 = tfg.layers.GCN(units=cfg.gnn.dim_inner, activation=tf.nn.relu)
        self.gcn1 = tfg.layers.GCN(units=cfg.gnn.dim_inner, activation=tf.nn.relu)
        self.gcn2 = tfg.layers.GCN(units=cfg.gnn.dim_inner, activation=tf.nn.relu)
        # After the GCN layers, we need to build an inference layer with "num_labels" outputs.
        self.mlp = tf.keras.models.Sequential([tf.keras.layers.Flatten()
                                               , tf.keras.layers.Dense(256, activation='relu')
                                               , tf.keras.layers.Dense(datasets[0].num_labels)])

    def call(self, inputs, training=None, mask=None, cache=None):
        x, edge_index, edge_weight = inputs  # extract the components from the input list.
        # Run through our series of GCN layers.
        h = self.gcn0([x, edge_index, edge_weight], cache=cache, training=training, mask=mask)
        h = self.gcn1([h, edge_index, edge_weight], cache=cache, training=training, mask=mask)
        h = self.gcn2([h, edge_index, edge_weight], cache=cache, training=training, mask=mask)
        # Run through the inference layer
        h = self.mlp(h)
        return h


class IDGCNModel(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Build out n specialized GCN layers.  In the paper, they used 3, so we simply hard coded that many layers.
        self.gcn0 = IDGCN(units=cfg.gnn.dim_inner, activation=tf.nn.relu)
        self.gcn1 = IDGCN(units=cfg.gnn.dim_inner, activation=tf.nn.relu)
        self.gcn2 = IDGCN(units=cfg.gnn.dim_inner, activation=tf.nn.relu)
        # After the specialized GCN layers, we need to build an inference layer with "num_labels" outputs.
        self.mlp = tf.keras.models.Sequential([tf.keras.layers.Flatten()
                                               , tf.keras.layers.Dense(256, activation='relu')
                                               , tf.keras.layers.Dense(datasets[0].num_labels)])

    def call(self, inputs, training=None, mask=None, cache=None):
        x, edge_index, id_index, edge_weight = inputs  # extract the components from the input list.
        # Run through our series of specialized GCN layers.
        # These layers require an additional 'id_index' input for processing
        h = self.gcn0([x, edge_index, id_index, edge_weight], cache=cache)  
        h = self.gcn1([h, edge_index, id_index, edge_weight], cache=cache)
        h = self.gcn2([h, edge_index, id_index, edge_weight], cache=cache)
        # Run through the inference layer
        h = self.mlp(h)
        return h
    
    
class GATModel(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Build out n GAT layers.  In the paper, they used 3, so we simply hard coded that many layers.
        self.gat0 = tfg.layers.GAT(units=cfg.gnn.dim_inner, activation=tf.nn.relu)
        self.gat1 = tfg.layers.GAT(units=cfg.gnn.dim_inner, activation=tf.nn.relu)
        self.gat2 = tfg.layers.GAT(units=cfg.gnn.dim_inner, activation=tf.nn.relu)
        # After the GAT layers, we need to build an inference layer with "num_labels" outputs.
        self.mlp = tf.keras.models.Sequential([tf.keras.layers.Flatten()
                                               , tf.keras.layers.Dense(256, activation='relu')
                                               , tf.keras.layers.Dense(datasets[0].num_labels)])

    def call(self, inputs, training=None, mask=None, cache=None):
        x, edge_index, _ = inputs  # extract the components from the input list.
        # Run through our series of GAT layers.
        h = self.gat0([x, edge_index], training=training, mask=mask)
        h = self.gat1([h, edge_index], training=training, mask=mask)
        h = self.gat2([h, edge_index], training=training, mask=mask)
        # Run through the inference layer
        h = self.mlp(h)
        return h


class IDGATModel(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Build out n specialized GAT layers.  In the paper, they used 3, so we simply hard coded that many layers.
        self.gat0 = IDGAT(units=cfg.gnn.dim_inner, activation=tf.nn.relu)
        self.gat1 = IDGAT(units=cfg.gnn.dim_inner, activation=tf.nn.relu)
        self.gat2 = IDGAT(units=cfg.gnn.dim_inner, activation=tf.nn.relu)
        # After the specialized GAT layers, we need to build an inference layer with "num_labels" outputs.
        self.mlp = tf.keras.models.Sequential([tf.keras.layers.Flatten()
                                               , tf.keras.layers.Dense(256, activation='relu')
                                               , tf.keras.layers.Dense(datasets[0].num_labels)])

    def call(self, inputs, training=None, mask=None, cache=None):
        x, edge_index, id_index, edge_weight = inputs  # extract the components from the input list.
        # Run through our series of specialized GAT layers.
        # These layers require an additional 'id_index' input for processing
        h = self.gat0([x, edge_index, id_index], training=training)  
        h = self.gat1([h, edge_index, id_index], training=training)
        h = self.gat2([h, edge_index, id_index], training=training)
        # Run through the inference layer
        h = self.mlp(h)
        return h    


class SAGEModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Build out n GraphSAGE layers.  In the paper, they used 3.
        self.sage_layers = []
        for _ in range(cfg.gnn.layers_mp):
            self.sage_layers.append(tfg.layers.MeanGraphSage(units=cfg.gnn.dim_inner, activation=tf.nn.relu))
        # After the GraphSAGE layers, we need to build an inference layer with "num_labels" outputs.
        self.mlp = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                               tf.keras.layers.Dense(256, activation='relu'),
                                               tf.keras.layers.Dense(datasets[0].num_labels)])

    def call(self, inputs, training=None, mask=None, cache=None):
        x, edge_index, _ = inputs  # extract the components from the input list.
        # Run through our series of GraphSAGE layers.
        h = x
        for layer in self.sage_layers:
            h = layer([h, edge_index], training=training, cache=cache, mask=mask)
        # Run through the inference layer
        h = self.mlp(h)
        return h


class IDSAGEModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Build out n specialized GraphSAGE layers.  In the paper, they used 3.
        self.sage_layers = []
        for _ in range(cfg.gnn.layers_mp):
            self.sage_layers.append(IDSAGE(units=cfg.gnn.dim_inner, activation=tf.nn.relu))
        # After the specialized GraphSAGE layers, we need to build an inference layer with "num_labels" outputs.
        self.mlp = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                               tf.keras.layers.Dense(256, activation='relu'),
                                               tf.keras.layers.Dense(datasets[0].num_labels)])

    def call(self, inputs, training=None, mask=None, cache=None):
        x, edge_index, id_index, _ = inputs  # extract the components from the input list.
        # Run through our series of specialized GraphSAGE layers.
        # These layers require an additional 'id_index' input for processing
        h = x
        for layer in self.sage_layers:
            h = layer([h, edge_index, id_index], training=training, cache=cache, mask=mask)
        # Run through the inference layer
        h = self.mlp(h)
        return h


class GINModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Build out n GIN layers.  In the paper, they used 3.
        self.gin_layers = []
        for _ in range(cfg.gnn.layers_mp):
            # The GIN layer must be passed a separate neural network model, which will learn the kernel to use.
            self.gin_layers.append(tfg.layers.GIN(
                mlp_model=tf.keras.Sequential([
                    tf.keras.layers.Dense(cfg.gnn.dim_inner, activation=tf.nn.relu),
                    tf.keras.layers.Dense(cfg.gnn.dim_inner),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation(tf.nn.relu)
                ])
            ))
        # After the GIN layers, we need to build an inference layer with "num_labels" outputs.
        self.mlp = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                               tf.keras.layers.Dense(256, activation='relu'),
                                               tf.keras.layers.Dense(datasets[0].num_labels)])

    def call(self, inputs, training=None, mask=None, cache=None):
        x, edge_index, edge_weight = inputs  # extract the components from the input list.
        # Run through our series of GIN layers.
        h = x
        for layer in self.gin_layers:
            h = layer([h, edge_index, edge_weight], training=training, cache=cache, mask=mask)
        # Run through the inference layer
        h = self.mlp(h)
        return h


class IDGINModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Build out n specialized GIN layers.  In the paper, they used 3.
        self.gin_layers = []
        for _ in range(cfg.gnn.layers_mp):
            # GIN models require a MLP layer to learn how to process the kernel.
            # The ID portion also requires its own MLP.
            self.gin_layers.append(
                IDGIN(
                    mlp_model=tf.keras.Sequential([  # GIN MLP
                        tf.keras.layers.Dense(cfg.gnn.dim_inner, activation=tf.nn.relu),
                        tf.keras.layers.Dense(cfg.gnn.dim_inner),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Activation(tf.nn.relu)
                    ]),
                    mlpid_model=tf.keras.Sequential([  # ID MLP
                        tf.keras.layers.Dense(cfg.gnn.dim_inner, activation=tf.nn.relu),
                        tf.keras.layers.Dense(cfg.gnn.dim_inner),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Activation(tf.nn.relu)
                    ])
                )
            )
        # After the specialized GIN layers, we need to build an inference layer with "num_labels" outputs.
        self.mlp = tf.keras.models.Sequential([tf.keras.layers.Flatten(),  # Run everything through a final inference layer
                                               tf.keras.layers.Dense(256, activation='relu'),
                                               tf.keras.layers.Dense(datasets[0].num_labels)])  # Classify based on labels

    def call(self, inputs, training=None, mask=None, cache=None):
        x, edge_index, id_index, edge_weight = inputs  # extract the components from the input list.
        # Run through our series of specialized GIN layers.
        # These layers require an additional 'id_index' input for processing
        h = x
        for layer in self.gin_layers:  # Iterate through each of the GIN layers
            h = layer([h, edge_index, id_index, edge_weight], training=training, mask=mask, cache=cache)
        # Run through the inference layer
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


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='ginconv')
args = parser.parse_args()
model = args.model

config_path = f'./config/{model}_tf'

# These are all of the datasets that we would like to evaluate against
datasets = ['CiteSeer', 'Cora', 'ENZYMES', 'PROTEINS', 'ws', 'ba']
# These are the tasks that we will be evaluating.
tasks = ['node']
files = [f'{model}_{task}_{dataset}' for dataset in datasets for task in tasks]  # generate a list of files to process

for config_name in files:  # iterate through each file for running different experiments
    acc_lists = []
    max_acc = []
    for i in range(repeat): # run each experiment "repeat" number of times
        # Load config file
        cfg.merge_from_file(config_path+'/'+config_name+'.yaml')
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
        model_func = {  # Select the model based on the content of the experiment's YAML file.
            'Tfg-idgcn': IDGCNModel,
            'Tfg-idsage': IDSAGEModel,
            'Tfg-idgat': IDGATModel,
            'Tfg-idgin': IDGINModel,
            'Tfg-gcnconv': GCNModel,
            'Tfg-gatconv': GATModel,
            'Tfg-sageconv': SAGEModel,
            'Tfg-ginconv': GINModel,
        }
        model = model_func[cfg.gnn.layer_type](datasets)
        #optimizer = create_optimizer(model.parameters())
        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.optim.base_lr)
        #print(cfg.optim.base_lr)
        #scheduler = create_scheduler(optimizer)
        #logging.info(model)
        #logging.info(cfg)
        cfg.params = 0

        if cfg.train.mode == 'standard':
            acc_list = train(meters, loaders, model, optimizer,datasets)
            acc_lists.append(acc_list)
            max_acc.append(max(acc_list))
        else:
            train_dict[cfg.train.mode](
                meters, loaders, model, optimizer, scheduler)
    # Write the results of the experiment to a file for sharing results.
    layer_type = f'id{cfg.gnn.layer_type}Fast' if cfg.dataset.augment_feature != [] else cfg.gnn.layer_type
    np.savetxt('./' + cfg.out_dir+'/val'+'/middle'+f'/{cfg.model.type}-{layer_type}'+f'_{cfg.dataset.name}.txt', np.array(acc_lists))
    np.savetxt('./' + cfg.out_dir+'/val'+'/final'+f'/{cfg.model.type}-{layer_type}'+f'_{cfg.dataset.name}_avg_acc.txt', np.array([np.mean(max_acc)]))
    print(f'The average validation accuracy of {repeat} rounds is: {np.mean(max_acc)}')

