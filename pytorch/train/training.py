#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: André Pacheco
E-mail: pacheco.comp@gmail.com

This file implements the CNN train phase

If you find any bug or have some suggestion, please, email me.
"""

import torch


def train_model (model, optimizer, loss_fn, data_loader, params):
    """

    :param model:
    :param optimizer:
    :param loss_fn:
    :param data:
    :param params:
    :return:
    """

    epochs = params['epochs']


    for epoch in range(epochs):

        for k, data in enumerate(data_loader):

            # Getting tthe input
            imgs_batch, labels_batch = data

            # Zero the parameters gradient
            optimizer.zero_grad()

            # Doing the forward pass
            out = model(imgs_batch)

            # Computing loss
            loss = loss_fn(out, labels_batch)

            # Computing gradients and computing the update step
            loss.backward()
            optimizer.step()


