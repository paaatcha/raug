#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: André Pacheco
E-mail: pacheco.comp@gmail.com

This file implements the CNN train phase

If you find any bug or have some suggestion, please, email me.
"""

import torch
from tqdm import tqdm
from ..model.checkpoints import load_model, save_model
from ..evaluate.eval import evaluate_model


def _train_epoch (model, optimizer, loss_fn, data_loader, params, c_epoch, t_epoch, device):
    """

    :param model:
    :param optimizer:
    :param loss_fn:
    :param data:
    :param params:
    :param epoch
    :return:
    """

    # epochs = params['epochs']
    has_extra_info = params['has_extra_info']

    # setting the model to training mode
    model.train()

    # Moving the model to the given device
    model.to(device)


    # Setting tqdm to show some information on the screen
    with tqdm(total=len(data_loader), ascii=True, desc='Epoch {}/{}: '.format(c_epoch, t_epoch), ncols=100) as t:

        loss_avg = 0
        # Getting the data from the DataLoader generator
        for batch, data in enumerate(data_loader, 1):

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
                # print (device)
                # exit()

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
            loss_avg = (loss_avg + loss.item()) / batch

            # Computing gradients and performing the update step
            loss.backward()
            optimizer.step()

            # Updating tqdm
            t.set_postfix(loss='{:05.3f}'.format(loss_avg))
            t.update()



def train_model (model, optimizer, loss_fn, train_data_loader, val_data_loader, params, save_folder=None,
                 saved_model=None):
    """

    :param model:
    :param optimizer:
    :param loss_fn:
    :param train_data_loader:
    :param val_data_loader:
    :param params:
    :param save_folder:
    :param saved_model:
    :return:
    """

    # Setting the device
    # If GPU is available, let's move the model to there
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(torch.cuda.current_device()))
    else:
        device = torch.device("cpu")

    # Checking if we have a saved model. If we have, load it, otherwise, let's train the model from scratch
    if (saved_model is not None):
        model = load_model(saved_model, model)

    # Checking if the most important parameters are set
    if ('epochs' not in params.keys()):
        raise Exception("You must inform 'epochs' in params dict")
    if ('has_extra_info' not in params.keys()):
        raise Exception("You must inform 'has_extra_info' in params dict")

    epochs = params['epochs']
    best_loss = 0
    best_acc = 999999
    best = False

    # Let's iterate for `epoch` epochs or a tolerance
    for epoch in range(epochs):

        _train_epoch(model, optimizer, loss_fn, train_data_loader, params, epoch, epoch, device)

        # After each epoch, we evaluate the model for the training and validation data
        val_metrics = evaluate_model (model, val_data_loader, loss_fn, device, 'Validation', True)

        if (val_metrics['accuracy'] > best_acc):
            print ('Hey, I found a new best accuracy: {}'.format(best_acc))
            best_acc = val_metrics['accuracy']
            best = True
        if (val_metrics['loss'] < best_loss):
            best_loss = val_metrics['loss']
            print('Hey, I found a new best loss: {}'.format(best_acc))
            best = True

        # Check if it's the best model in order to save it
        if (save_folder is not None):
            save_model(model, save_folder, epoch, best)

        # TODO the LOGGER to tensorboard
        best = False






