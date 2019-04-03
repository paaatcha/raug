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

def save_model (model, folder_path, epoch, is_best):
    """
    This function saves the parameters of a model. It saves the last and best model (if it's the best).

    :param model (nn.Model): the model you wanna save the parameters
    :param folder_path (string): the folder you wanna save the checkpoints
    :param name_last (string): the file's name of the last checkpoint. Considers using epoch in the name
    :param name_best (bool, optional): the file's name of the best checkpoint. If it's False, it means this checkpoint
    is not the best one. Default is false.
    """

    last_check_path = os.path.join(folder_path, 'last-checkpoint')
    best_check_path = os.path.join(folder_path, 'best-checkpoint')

    if (os.path.exists(last_check_path)):
        print ('last-checkpoint folder does not exist. I am creating it!')
        os.mkdir(last_check_path)
    else:
        print ('last-checkpoint folder exist! Perfect, I will just use it.')

    if (os.path.exists(best_check_path)):
        print('best-checkpoint folder does not exist. I am creating it!')
        os.mkdir(best_check_path)
    else:
        print('best-checkpoint folder exist! Perfect, I will just use it.')

    torch.save(model.state_dict(), os.path.join(last_check_path, "last-checkpoint.pht"))
    with open(os.path.join(last_check_path, "last-epoch"), "w") as f:
        f.write(epoch)

    if (is_best):
        torch.save(model.state_dict(), os.path.join(best_check_path, 'best-checkpoint-{}.pth'.format(epoch)))


def load_model (checkpoint_path, model):
    """
    This function loads a model from a given checkpoint.

    :param checkpoint_path (string): the full path to de checkpoint
    :param model (nn.Model): the model that you wanna load the parameters
    :return (nn.Model): the loaded model
    """

    if (not os.path.exists(checkpoint_path)):
        raise ("The {} does not exist!".format(checkpoint_path))

    model.load_state_dict(torch.load(checkpoint_path))

    return model
