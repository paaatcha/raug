#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: André Pacheco
E-mail: pacheco.comp@gmail.com

This file implements the CNN train phase

If you find any bug or have some suggestion, please, email me.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from ..model.checkpoints import load_model, save_model
from ..evaluate.eval import evaluate_model
from tensorboardX import SummaryWriter
import numpy as np


def _train_epoch (model, optimizer, loss_fn, data_loader, c_epoch, t_epoch, device):
    """
    This function trains an epoch of the dataset, that is, it goes through all dataset batches once.
    :param model (torch.nn.Module): a module to be trained
    :param optimizer (torch.optim.optim): an optimizer to fit the model
    :param loss_fn (torch.nn.Loss): a loss function to evaluate the model prediction
    :param data_loader (torch.utils.DataLoader): a dataloader containing the dataset
    :param c_epoch (int): the current epoch
    :param t_epoch (int): the total number of epochs
    :param device (torch.device): the device to carry out the training 
    """

    # setting the model to training mode
    model.train()

    # Setting tqdm to show some information on the screen
    with tqdm(total=len(data_loader), ascii=True, desc='Epoch {}/{}: '.format(c_epoch+1, t_epoch), ncols=100) as t:

        loss_avg = 0
        # Getting the data from the DataLoader generator
        for batch, data in enumerate(data_loader, 0):

            # In data we may have imgs, labels and extra info. If extra info is [], it means we don't have it
            # for the this training case. Imgs came in data[0], labels in data[1] and extra info in data[2]
            imgs_batch, labels_batch, extra_info_batch = data
            if (len(extra_info_batch)):
                # In this case we have extra information and we need to pass this data to the model
                # Moving the data to the deviced that we set above
                imgs_batch, labels_batch = imgs_batch.to(device), labels_batch.to(device)
                extra_info_batch = extra_info_batch.to(device)

                # Doing the forward pass
                out = model(imgs_batch, extra_info_batch)
            else:
                # In this case we don't have extra info, so the model doesn't expect for it
                # Moving the data to the deviced that we set above
                imgs_batch, labels_batch = imgs_batch.to(device), labels_batch.to(device)

                # Doing the forward pass
                out = model(imgs_batch)

            # Computing loss function
            loss = loss_fn(out, labels_batch)
            loss_avg += loss.item()

            # Zero the parameters gradient
            optimizer.zero_grad()

            # Computing gradients and performing the update step
            loss.backward()
            optimizer.step()

            # Updating tqdm
            t.set_postfix(loss='{:05.3f}'.format(loss_avg/(batch+1)))
            t.update()



def train_model (model, train_data_loader, val_data_loader, optimizer=None, loss_fn=None, epochs=10,
                 epochs_early_stop=None, save_folder=None, saved_model=None, class_names=None,
                 best_metric="loss", metrics=["accuracy"], metrics_options=None, device=None):
    """
    This is the main function to carry out the training phase.

    :param c_epoch (int): the current epoch
    :param t_epoch (int): the total number of epochs
    :param device (torch.device): the device to carry out the training
    :param model (torch.nn.Module): a module to be trained
    :param train_data_loader (torch.utils.DataLoader): a dataloader containing the train dataset
    :param val_data_loader (torch.utils.DataLoader): a dataloader containing the validation dataset
    :param optimizer (torch.optim.optim, optional): an optimizer to fit the model. If None, it will use the
     optim.Adam(model.parameters(), lr=0.001). Default is None.
    :param loss_fn (torch.nn.Loss, optional): a loss function to evaluate the model prediction. If None, it will use the
    nn.CrossEntropyLoss(). Default is None.
    :param epochs (int, optional): the number of epochs to train the model. Default is 10.
    :param epochs_early_stop (int, optional): if you'd like to check early stop, pass the number of epochs that need to
    be achieved to stop the training. It checks if the loss is improving. If it doesn't improve for epochs_early_stop,
    training stops. If None, the training is never stopped. Default is None.
    :param save_folder (string, optional): if you'd like to save the last and best checkpoints, just pass the folder
    path in which the checkpoint will be saved. If None, the model is not saved in the disk. Default is None.
    :param saved_model (string, optinal): if you'd like to restart the training from a given saved checkpoint, pass
    the path to this file here. If None, the model starts training from scratch. Default is None.
    :param class_names (list, optional): the list of class names.
    :param best_metric (string, optional): if you chose save the model, you can inform the metric you'd like to save as
    the best. Default is loss.
    :param metrics (list, optional): a list containing the metrics you'd like to compute after every epoch. To check the
    available metrics, please refers to jedy.pytorch.model.metrics. Default is only accuracy
    :param metrics_options (dict, optional): options to compute the metrics. For more information, please refers to
    jedy.pytorch.model.metrics. Default is only accuracy. Default is None.
    :param device (torch.device): the device you'd like to train the model. If None, it will check if you have a GPU
    available. If not, it use the CPU. Default is None.
    :return: 
    """


    if (loss_fn is None):
        loss_fn = nn.CrossEntropyLoss()

    if (optimizer is None):
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Setting the device
    # If GPU is available, let's move the model to there
    if (device is None):
        if torch.cuda.is_available():

            device = torch.device("cuda")
            # device = torch.device("cuda:" + str(torch.cuda.current_device()))
            m_gpu = torch.cuda.device_count()
            if (m_gpu > 1):
                print ("The training will be carry out using {} GPUs".format(m_gpu))
                model = nn.DataParallel(model)
            else:
                print("The training will be carry out using 1 GPU")
        else:
            print("The training will be carry out using CPU")
            device = torch.device("cpu")

    # Moving the model to the given device
    model.to(device)

    # Checking if we have a saved model. If we have, load it, otherwise, let's train the model from scratch
    if (saved_model is not None):
        model = load_model(saved_model, model)

    # Setting data to store the best mestric
    if (best_metric is 'loss'):
        best_metric_value = np.inf
    else:
        best_metric_value = 0
    best_flag = False

    # setting a flag for the early stop
    early_stop_count = 0
    best_val_loss = np.inf

    # writer is used to generate the summary files to be loaded at tensorboard
    writer = SummaryWriter (os.path.join(save_folder, 'summaries'))

    # Let's iterate for `epoch` epochs or a tolerance
    for epoch in range(epochs):

        _train_epoch(model, optimizer, loss_fn, train_data_loader, epoch, epochs, device)

        # After each epoch, we evaluate the model for the training and validation data
        train_metrics = evaluate_model(model, train_data_loader, loss_fn=loss_fn, device=device, partition_name='Train',
                                       metrics=["accuracy"], verbose=True)

        print ('\n')

        # After each epoch, we evaluate the model for the training and validation data
        val_metrics = evaluate_model (model, val_data_loader, loss_fn=loss_fn, device=device,
                    partition_name='Validation', metrics=metrics, class_names=class_names,
                                      metrics_options=metrics_options, verbose=True)

        # writer.add_scalar('Accuracy', train_metrics['accuracy'], epoch)
        # writer.add_scalar('val/accuracy', val_metrics['accuracy'], epoch)
        #
        # writer.add_scalar('Loss', train_metrics['loss'], epoch)
        # writer.add_scalar('val/loss', val_metrics['loss'], epoch)


        writer.add_scalars('Loss', {'val-loss': val_metrics['loss'],
                                                 'train-loss': train_metrics['loss']},
                                                 epoch)

        writer.add_scalars('Accuracy', {'val-loss': val_metrics['accuracy'],
                                    'train-loss': train_metrics['accuracy']},
                                    epoch)

        if (best_metric is 'loss'):
            if (val_metrics[best_metric] < best_metric_value):
                best_metric_value = val_metrics[best_metric]
                print('- New best {}: {:.3f}'.format(best_metric, best_metric_value))
                best_flag = True
        else:
            if (val_metrics[best_metric] > best_metric_value):
                best_metric_value = val_metrics[best_metric]
                print('- New best {}: {:.3f}'.format(best_metric, best_metric_value))
                best_flag = True

        # Check if it's the best model in order to save it
        if (save_folder is not None):
            print ('- Saving the model...\n')
            save_model(model, save_folder, epoch, best_flag)
        
        best = False

        # Cheking if the validation loss has improved
        if (epochs_early_stop is not None):
            val_loss = val_metrics['loss']

            if (val_loss < best_val_loss):
                best_val_loss = val_loss
                early_stop_count = 0
            else:
                early_stop_count+=1;

            if (early_stop_count >= epochs_early_stop):
                print ("The early stop trigger was activated. The validation loss " +
                       "{:.3f} did not improved for {} epochs.".format(best_val_loss, epochs_early_stop) +
                       "The training phase was stopped.")

                break


    writer.close()






