import torch
import time
import logging

from graphgym.config import cfg
from graphgym.loss import compute_loss, compute_loss_Tfg
from graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch
from graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt


from tf_geometric.data.graph import Graph
import tensorflow as tf


def train_epoch(logger, loader, model, optimizer):
    model.train()
    time_start = time.time()
    for batch in loader:
        print(batch)
        optimizer.zero_grad()
        batch.to(torch.device(cfg.device))
        pred, true = model(batch)
        loss, pred_score = compute_loss(pred, true)
        loss.backward()
        optimizer.step()
        logger.update_stats(true=true.detach().cpu(),
                            pred=pred_score.detach().cpu(),
                            loss=loss.item(),
                            lr=cfg.optim.base_lr,
                            time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()
   # scheduler.step()


def train_epoch_Tfg(logger, loader, model, optimizer, datasets):
    loss_s = 0
    time_start = time.time()
    for batch in loader:  # Take portions of the dataset to perform training
        if cfg.dataset.augment_feature != []:  # If our YAML defines an augmented feature
            # Generate our node features by concatenating
            # Build the graph for training using node_feature, edge_index and node_label
            node_feature = torch.cat((batch[cfg.dataset.augment_feature[0]], batch['node_feature']), 1)
            graph = Graph(x=node_feature.numpy(), edge_index=batch.edge_index.numpy(), y=batch.node_label.numpy())
        else:  # If our YAML does not have augmented features defined:
            graph = Graph(x=batch.node_feature.numpy(), edge_index=batch.edge_index.numpy(), y=batch.node_label.numpy())
        with tf.GradientTape() as tape:
            if 'id' in cfg.gnn.layer_type:  # if we're using the full ID-GNN model, include node_id_index as an input.
                logits = model([graph.x, graph.edge_index, batch.node_id_index.numpy(), graph.edge_weight], training=True)
            else:
                logits = model([graph.x, graph.edge_index, graph.edge_weight], training=True)
            # Compute the loss of the classifier
            loss = compute_loss_Tfg(logits, batch.node_label_index, batch.node_label, tape.watched_variables(), datasets)
            vars = tape.watched_variables()  # Retrieve a list of the trainable parameters
            grads = tape.gradient(loss, vars)  # Retrieve the gradients for each of the trainable parameters
            optimizer.apply_gradients(zip(grads, vars))  # Apply the gradients
        logger.update_stats(true=torch.tensor([1]*128),  # Save progress in a logger instance.
                            pred=torch.tensor([1]*128),
                            loss=loss,
                            lr=cfg.optim.base_lr,
                            time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()


@torch.no_grad()
def eval_epoch(logger, loader, model):
    model.eval()
    time_start = time.time()
    for batch in loader:
        batch.to(torch.device(cfg.device))
        pred, true = model(batch)
        loss, pred_score = compute_loss(pred, true)
        logger.update_stats(true=true.detach().cpu(),
                            pred=pred_score.detach().cpu(),
                            loss=loss.item(),
                            lr=0,
                            time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()
        

def eval_epoch_Tfg(loader, model):
    accuracy_sum = 0
    for batch in loader:  # Take a portion of the dataset for evaluation
        if cfg.dataset.augment_feature != []:
            node_feature = torch.cat((batch[cfg.dataset.augment_feature[0]], batch['node_feature']), 1)
            graph = Graph(x=node_feature.numpy(), edge_index=batch.edge_index.numpy(), y=batch.node_label.numpy())
        else:
            graph = Graph(x=batch.node_feature.numpy(), edge_index=batch.edge_index.numpy(), y=batch.node_label.numpy())    
        if 'id' in cfg.gnn.layer_type:    
            logits = model([graph.x, graph.edge_index, batch.node_id_index.numpy(), graph.edge_weight], training=False)
        else:
            logits = model([graph.x, graph.edge_index, graph.edge_weight], training=False)
        # Everything above this line is the same as the training step before the gradient tape.
        masked_logits = tf.gather(logits, batch.node_label_index)  # Infer values for the predicted classes
        y_pred = tf.argmax(masked_logits, axis=-1, output_type=tf.int32)  # Take the highest value as the likely class
        if cfg.dataset.transform == 'ego':
            # For Full ID-GNN evaluation, gathering the required test node labels indexed by node_label_index from all the node labels
            masked_labels = tf.cast(tf.gather(batch.node_label, batch.node_label_index), y_pred.dtype)
        else:
            masked_labels = tf.cast(batch.node_label, y_pred.dtype)
        
        corrects = tf.equal(y_pred, masked_labels)  # See if the predictions match the labels
        accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))  # Calculate the accuracy of the predictions
        accuracy_sum += accuracy
    accuracy = accuracy_sum / len(loader)  # Calculate the average of all the batches accuracies
    return accuracy
        

'''
remove scheduler
'''


def train(loggers, loaders, model, optimizer, datasets):
    start_epoch = 0
    valid_acc_list = []
    #if cfg.train.auto_resume:
    #    start_epoch = load_ckpt(model, optimizer, scheduler)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch {}'.format(start_epoch))

    num_splits = len(loggers)
    print(num_splits)
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        if not cfg.gnn.layer_type[:3] == 'Tfg':
            train_epoch(loggers[0], loaders[0], model, optimizer)
        else:
            train_epoch_Tfg(loggers[0], loaders[0], model, optimizer, datasets)
        #loggers[0].write_epoch(cur_epoch)
        if is_eval_epoch(cur_epoch):
            valid_acc_list_ = []
            for i in range(1, num_splits):
                if not cfg.gnn.layer_type[:3] == 'Tfg':
                    eval_epoch(loggers[i], loaders[i], model)
                else:
                    valid_acc = eval_epoch_Tfg(loaders[i], model).numpy()
                    valid_acc_list_.append(valid_acc)
            valid_acc_list.append(sum(valid_acc_list_)/len(valid_acc_list_))
            print(f'epoch {cur_epoch}, acc:{sum(valid_acc_list_)/len(valid_acc_list_)}')
                
                #loggers[i].write_epoch(cur_epoch)
        #if is_ckpt_epoch(cur_epoch):
        #    save_ckpt(model, optimizer, scheduler, cur_epoch)
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()
    logging.info('Task done, results saved in {}'.format(cfg.out_dir))
    print(f'The best validation accuracy is {max(valid_acc_list)}, the epoch is {10*valid_acc_list.index(max(valid_acc_list))}')
    return valid_acc_list

