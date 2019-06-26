#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: André Pacheco
E-mail: pacheco.comp@gmail.com

This file contains the functions used to save and load trained model

If you find any bug or have some suggestion, please, email me.
"""

import torch
import os
import torch.nn as nn

def save_model (model, folder_path, epoch, is_best, verbose=False):
    """
    This function saves the parameters of a model. It saves the last and best model (if it's the best).

    :param model (nn.Model): the model you wanna save the parameters
    :param folder_path (string): the folder you wanna save the checkpoints
    :param name_last (string): the file's name of the last checkpoint. Considers using epoch in the name
    :param name_best (bool, optional): the file's name of the best checkpoint. If it's False, it means this checkpoint
    :param verbose (bool, optional): If you'd like to print information on the screen. Default is False.
    is not the best one. Default is false.
    """

    last_check_path = os.path.join(folder_path, 'last-checkpoint')
    best_check_path = os.path.join(folder_path, 'best-checkpoint')

    # print(last_check_path)
    # print(os.path.exists(last_check_path))
    # exit()

    if (not os.path.exists(last_check_path)):
        if (verbose):
            print ('last-checkpoint folder does not exist. I am creating it!')
        os.mkdir(last_check_path)
    else:
        if (verbose):
            print ('last-checkpoint folder exist! Perfect, I will just use it.')

    if (not os.path.exists(best_check_path)):
        if (verbose):
            print('best-checkpoint folder does not exist. I am creating it!')
        os.mkdir(best_check_path)
    else:
        if (verbose):
            print('best-checkpoint folder exist! Perfect, I will just use it.')

    torch.save(model.state_dict(), os.path.join(last_check_path, "last-checkpoint.pht"))
    with open(os.path.join(last_check_path, "last-epoch"), "w") as f:
        f.write('EPOCH: {}'.format(epoch))

    if (is_best):
        torch.save(model.state_dict(), os.path.join(best_check_path, 'best-checkpoint.pth'))


def load_model (checkpoint_path, model):
    """
    This function loads a model from a given checkpoint.

    :param checkpoint_path (string): the full path to de checkpoint
    :param model (nn.Model): the model that you wanna load the parameters
    :return (nn.Model): the loaded model
    """

    if (not os.path.exists(checkpoint_path)):
        raise Exception ("The {} does not exist!".format(checkpoint_path))

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    model.load_state_dict(torch.load(checkpoint_path))

    return model

