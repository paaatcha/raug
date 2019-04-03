#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: André Pacheco
E-mail: pacheco.comp@gmail.com

This file function to evaluate a model

If you find any bug or have some suggestion, please, email me.
"""

import torch
from ..model.metrics import accuracy

# Evaluate the model and the validation
def evaluate_model (model, data_loader, loss_fn, device, partition_name='Val', verbose=True):
    """
    This function evaluates a given model for a fiven data_loader

    :param model (nn.Model): the model you'd like to evaluate
    :param data_loader (DataLoader): the DataLoader containing the data partition
    :param loss_fn (nn.Loss): the loss function used in the training
    :param partition_name (string): the partition name
    :param device (torch.device, optional): the device to use. Default is 'cpu'
    :param verbose (bool, optional): if you'd like to print information o the screen. Default is True

    :return: a dictionary containing the metrics
    """

    # setting the model to evaluation mode
    model.eval()

    # Moving the model to the device
    model.to(device)

    # Setting require_grad=False in order to dimiss the gradient computation in the graph
    with torch.no_grad():

        n_samples = len(data_loader)
        acc_avg = 0
        loss_avg = 0
        for data in data_loader:
            images_batch, labels_batch = data
            images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)
            out = model(images_batch)
            L = loss_fn(out, labels_batch)

            acc_avg += accuracy(out, labels_batch)
            loss_avg += L

        acc_avg = acc_avg / n_samples
        loss_avg = loss_avg / n_samples

    if (verbose):
        print('\n### {} metrics ###:'.format(partition_name))
        print('- Accuracy: {}'.format(acc_avg))
        print ('- Loss: {}'.format(loss_avg))
        print ('###################\n')

    return {
        "accuracy": acc_avg,
        "loss": loss_avg
    }