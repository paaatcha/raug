#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: André Pacheco
E-mail: pacheco.comp@gmail.com

This file function to evaluate a model

If you find any bug or have some suggestion, please, email me.
"""

import torch
import torch.nn as nn
import numpy as np
from ..model.metrics import accuracy, conf_matrix, plot_conf_matrix, precision_recall_report
from ..model.checkpoints import load_model

# Evaluate the model and the validation
def evaluate_model (model, data_loader, checkpoint_path= None, loss_fn=None, device=None,
                    partition_name='Eval', verbose=True):
    """
    This function evaluates a given model for a fiven data_loader

    :param model (nn.Model): the model you'd like to evaluate
    :param data_loader (DataLoader): the DataLoader containing the data partition
    :param checkpoint_path(string, optional): string with a checkpoint to load the model. If None, none checkpoint is
    loaded. Default is None.
    :param loss_fn (nn.Loss): the loss function used in the training
    :param partition_name (string): the partition name
    :param device (torch.device, optional): the device to use. If None, the code will look for a device. Default is None.
    :param verbose (bool, optional): if you'd like to print information o the screen. Default is True

    :return: a dictionary containing the metrics
    """

    if (checkpoint_path is not None):
        model = load_model(checkpoint_path, model)

    # setting the model to evaluation mode
    model.eval()

    if (device is None):
        # Setting the device
        if torch.cuda.is_available():
            device = torch.device("cuda:" + str(torch.cuda.current_device()))
        else:
            device = torch.device("cpu")

    # Moving the model to the given device
    model.to(device)

    if (loss_fn is None):
        loss_fn = nn.CrossEntropyLoss()

    # Setting require_grad=False in order to dimiss the gradient computation in the graph
    with torch.no_grad():

        n_samples = len(data_loader)
        acc_avg = 0
        loss_avg = 0
        all_preds = np.array([])
        all_labels = np.array([])

        for data in data_loader:
            images_batch, labels_batch = data
            images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)
            out = model(images_batch)
            L = loss_fn(out, labels_batch)

            # Getting the values as number instead of a prob vector
            _, pred = torch.max(out.data, 1)

            # Moving the data to CPU and converting it to numpy in order to it in scikit-learn
            pred_np = pred.cpu().numpy()
            labels_batch_np = labels_batch.cpu().numpy()

            # TODO setar as metricas que deseja investigar
            # if (conf):
            all_preds = np.concatenate((all_preds, pred_np))
            all_labels = np.concatenate((all_labels, labels_batch_np))


            acc_avg += accuracy(labels_batch_np, pred_np)
            loss_avg += L

        cm = conf_matrix(all_labels, all_preds)
        print (cm)
        # plot_conf_matrix (cm, ['0', '1', '2', '3', '4', '5'], normalize=True)


        # print (all_labels)
        # print (all_preds)

        precision_recall_report(all_labels, all_preds, True)

        # print (all_preds)
        # print (all_labels)

        acc_avg = acc_avg / n_samples
        loss_avg = loss_avg / n_samples

    if (verbose):
        print('- {} metrics:'.format(partition_name))
        print('- Accuracy: {:.3f}'.format(acc_avg))
        print ('- Loss: {:.3f}'.format(loss_avg))

    return {
        "accuracy": acc_avg,
        "loss": loss_avg
    }

