#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: André Pacheco
E-mail: pacheco.comp@gmail.com

This file function to evaluate a model

If you find any bug or have some suggestion, please, email me.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from ..model.metrics import Metrics
from ..model.checkpoints import load_model
from ...utils import common

# Evaluate the model and the validation
def evaluate_model (model, data_loader, checkpoint_path= None, loss_fn=None, device=None,
                    partition_name='Eval', metrics=['accuracy'], class_names=None, metrics_options=None,
                    has_extra_info=False, verbose=True):
    """
    This function evaluates a given model for a fiven data_loader

    :param model (nn.Model): the model you'd like to evaluate
    :param data_loader (DataLoader): the DataLoader containing the data partition
    :param checkpoint_path(string, optional): string with a checkpoint to load the model. If None, none checkpoint is
    loaded. Default is None.
    :param loss_fn (nn.Loss): the loss function used in the training
    :param partition_name (string): the partition name
    :param metrics (list, tuple or string): it's the variable that receives the metrics you'd like to compute. Default
        is only the accuracy.
    :param class_names (list, tuple): a list or tuple containing the classes names in the same order you use in the
        label. For ex: ['C1', 'C2']. For more information about the options, please, refers to
        jedy.pytorch.model.metrics.py
    :param metrics_ options (dict): this is a dict containing some options to compute the metrics. Default is None.
    For more information about the options, please, refers to jedy.pytorch.model.metrics.py
    :param device (torch.device, optional): the device to use. If None, the code will look for a device. Default is
    None. For more information about the options, please, refers to jedy.pytorch.model.metrics.py
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
        loss_avg = 0

        # Setting the metrics object
        metrics = Metrics (metrics, class_names, metrics_options)

        for data in data_loader:

            # Getting the data batch considering if we have the extra information
            if (has_extra_info):
                images_batch, labels_batch, extra_info_batch = data
                # Moving the data to the deviced that we set above
                images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)
                extra_info_batch = extra_info_batch.to(device)
                pred_batch = model(images_batch, extra_info_batch)
            else:
                # Moving the data to the deviced that we set above
                images_batch, labels_batch = data
                images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)
                pred_batch = model(images_batch)

            # Computing the loss
            L = loss_fn(pred_batch, labels_batch)
            loss_avg += L

            # Moving the data to CPU and converting it to numpy in order to compute the metrics
            pred_batch_np = pred_batch.cpu().numpy()
            labels_batch_np = labels_batch.cpu().numpy()

            # updating the scores
            metrics.update_scores(labels_batch_np, pred_batch_np)

        # Getting the loss average
        loss_avg = loss_avg / n_samples

        # Getting the metrics
        metrics.compute_metrics()

        # Adding loss into the metric values
        metrics.add_metric_value("loss", loss_avg)

    if (verbose):
        print('- {} metrics:'.format(partition_name))
        metrics.print()

    return metrics.metrics_values


def visualize_model (model, data_loader, class_names, n_imgs=8, checkpoint_path= None, device_name="cpu",
                     has_extra_info=False, save_path=None):

    # Loading the model
    if (checkpoint_path is not None):
        model = load_model(checkpoint_path, model)

    # setting the model to evaluation mode
    model.eval()

    # setting the device
    device = torch.device(device_name)

    # Moving the model to the given device
    model.to(device)

    # If it's None, we're going to run for the whole data_loader
    if (n_imgs == 'all'):
        n_imgs = len(data_loader) * data_loader.batch_size

    plotted_imgs = 0

    # Setting require_grad=False in order to dimiss the gradient computation in the graph
    with torch.no_grad():

        for data in data_loader:

            # Getting one batch considering if we have the extra information
            if (has_extra_info):
                images_batch, labels_batch, extra_info_batch = data
                # Moving the data to the deviced that we set above
                images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)
                extra_info_batch = extra_info_batch.to(device)
                pred_batch = model(images_batch, extra_info_batch)
            else:
                # Moving the data to the deviced that we set above
                images_batch, labels_batch = data
                images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)
                pred_batch = model(images_batch)

            # Getting the label
            _, pred = torch.max(pred_batch.data, 1)

            if (device_name is not "cpu"):
                # Moving the data to CPU and converting it to numpy in order to compute the metrics
                pred_batch_np = pred.cpu().numpy()
                labels_batch_np = labels_batch.cpu().numpy()
                images_batch_np = images_batch.cpu().numpy()
            else:
                pred_batch_np = pred.numpy()
                labels_batch_np = labels_batch.numpy()
                images_batch_np = images_batch.numpy()

            # Getting only the number of images
            if (n_imgs is not None):
                pred_batch_np = pred_batch_np[0:n_imgs]
                labels_batch_np = labels_batch_np[0:n_imgs]
                images_batch_np = images_batch[0:n_imgs, :, :, :]

            for k in range(pred_batch_np.shape[0]):
                title = "real: {} - pred: {}".format(class_names[pred_batch_np[k]], class_names[labels_batch_np[k]])
                hit = pred_batch_np[k] == labels_batch_np[k]

                img_path_name = os.path.join(save_path, "img_{}.png".format(plotted_imgs))
                common.plot_img(images_batch_np[k], title=title, hit=hit, save_path=img_path_name)
                plotted_imgs +=1
                print ("Plotting image {} ...".format(plotted_imgs))

            if (plotted_imgs >= n_imgs):
                break






