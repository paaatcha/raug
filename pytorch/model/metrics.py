#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: André Pacheco
E-mail: pacheco.comp@gmail.com

This file implements the metrics to be used in the evaluation

If you find any bug or have some suggestion, please, email me.
"""

import torch

def accuracy (outputs, labels, verbose=False):
    """
    Just a simple functin to compute the accuracy

    :param outputs: the predictions returned by the model
    :param labels: the data real labels
    :param verbose: if you'd like to print the accuracy. Dafault is False.
    :return: the accuracy
    """


    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels).sum().item()
    acc = correct / outputs.shape[0]

    if (verbose):
        print('Accuracy - {}'.format(acc))

    return acc

