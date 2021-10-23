import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import tensorflow.keras.backend as K

from graphgym.contrib.loss import *
import graphgym.register as register
from graphgym.config import cfg


def compute_loss(pred, true, watched_vars=None):
    """
    TODO: Convert to tensorflow (high priority)- JB
    """
    '''

    :param pred: unnormalized prediction
    :param true: label
    :return: loss, normalized prediction score
    '''
    if cfg.dataset.format == 'TfG':
        bce_loss = tf.keras.losses.BinaryCrossEntropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        mse_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    else:
        bce_loss = nn.BCEWithLogitsLoss(reduction=cfg.model.size_average)
        mse_loss = nn.MSELoss(reduction=cfg.model.size_average)

    # default manipulation for pred and true
    # can be skipped if special loss computation is needed
    # if multi task binary classification, treat as flatten binary
    pred = pred.squeeze(-1) if pred.ndim > 1 else pred
    true = true.squeeze(-1) if true.ndim > 1 else true
    if true.ndim > 1 and cfg.model.loss_fun == 'cross_entropy':
        if cfg.dataset.format == 'TfG':
            pred, true = tf.reshape(pred, [-1]), tf.reshape(true, [-1])
        else:
            pred, true = torch.flatten(pred), torch.flatten(true)

    # Try to load customized loss
    for func in register.loss_dict.values():
        value = func(pred, true)
        if value is not None:
            return value

    if cfg.model.loss_fun == 'cross_entropy':
        # multiclass
        if pred.ndim > 1:
            if cfg.dataset.format == 'TfG':
                pred = K.log(K.softmax(pred, axis=-1))
                return K.categorical_crossentropy(true, pred), pred
            else:
                pred = F.log_softmax(pred, dim=-1)
                return F.nll_loss(pred, true), pred
        # binary
        else:
            if cfg.dataset.format == 'TfG':
                true = tf.cast(true, tf.float32)
                return bce_loss(true, pred), tf.sigmoid(pred)
            else:
                true = true.float()
                return bce_loss(pred, true), torch.sigmoid(pred)
    elif cfg.model.loss_fun == 'mse':
        if cfg.dataset.format == 'TfG':
            true = tf.cast(true, tf.float32)
            return mse_loss(true, pred), pred
        else:
            true = true.float()
            return mse_loss(pred, true), pred
    else:
        raise ValueError('Loss func {} not supported'.
                         format(cfg.model.loss_fun))
