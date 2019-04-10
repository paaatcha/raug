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


def _train_epoch (model, optimizer, loss_fn, data_loader, c_epoch, t_epoch, device, has_extra_info):
    """
    This function performs the training phase for a batch of data
    :param model:
    :param optimizer:
    :param loss_fn:
    :param data:
    :param params:
    :param epoch
    """

    # setting the model to training mode
    model.train()

    # Setting tqdm to show some information on the screen
    with tqdm(total=len(data_loader), ascii=True, desc='Epoch {}/{}: '.format(c_epoch+1, t_epoch), ncols=100) as t:

        loss_avg = 0
        # Getting the data from the DataLoader generator
        for batch, data in enumerate(data_loader, 0):

            # Getting the data batch considering if we have the extra information
            if (has_extra_info):
                imgs_batch, labels_batch, extra_info_batch = data
                # Moving the data to the deviced that we set above
                imgs_batch, labels_batch = imgs_batch.to(device), labels_batch.to(device)
                extra_info_batch = extra_info_batch.to(device)
            else:
                imgs_batch, labels_batch = data
                # Moving the data to the deviced that we set above
                imgs_batch, labels_batch = imgs_batch.to(device), labels_batch.to(device)

            # Zero the parameters gradient
            optimizer.zero_grad()

            # Doing the forward pass. If we have extra information we need to pass it to the model. So, the model must
            # expect this parameter
            if (has_extra_info):
                out = model (imgs_batch, extra_info_batch)
            else:
                out = model(imgs_batch)

            # Computing loss function
            loss = loss_fn(out, labels_batch)
            loss_avg += loss.item()


            # Computing gradients and performing the update step
            loss.backward()
            optimizer.step()

            # Updating tqdm
            t.set_postfix(loss='{:05.3f}'.format(loss_avg/(batch+1)))
            t.update()



def train_model (model, train_data_loader, val_data_loader, optimizer=None, loss_fn=None, epochs=10, 
                 has_extra_info=False, save_folder=None, saved_model=None, class_names=None, 
                 best_metric="accuracy", metrics=["accuracy"], metrics_options=None, device=None):
    """
    Function to train a given model.
    :param model (torch.nn.Model): a given model to train
    :param optimizer:
    :param loss_fn:
    :param train_data_loader:
    :param val_data_loader:
    :param params:
    :param save_folder:
    :param saved_model:
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
        best_metric_value = 999999
    else:
        best_metric_value = 0
    best_flag = False

    # writer is used to generate the summary files to be loaded at tensorboard
    writer = SummaryWriter (os.path.join(save_folder, 'summaries'))

    # Let's iterate for `epoch` epochs or a tolerance
    for epoch in range(epochs):

        _train_epoch(model, optimizer, loss_fn, train_data_loader, epoch, epochs, device, has_extra_info)

        # After each epoch, we evaluate the model for the training and validation data
        train_metrics = evaluate_model(model, train_data_loader, loss_fn=loss_fn, device=device,
                                     partition_name='Train', metrics=["accuracy"], verbose=True)

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

        if (val_metrics[best_metric] > best_metric_value):
            best_metric_value = val_metrics[best_metric]
            print('- New best {}: {}'.format(best_metric, best_metric_value))
            best_flag = True

        # Check if it's the best model in order to save it
        if (save_folder is not None):
            print ('- Saving the model...\n')
            save_model(model, save_folder, epoch, best_flag)
        
        best = False

    writer.close()






