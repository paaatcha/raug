#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: André Pacheco
E-mail: pacheco.comp@gmail.com

This file implements different networks that you can use for your task.

If you find any bug or have some suggestion, please, email me.
"""

import torch.nn as nn
import torch.nn.functional as F


class SingsNet (nn.Module):
    # Vai entrar um imagem (N, 3, 64, 64)
    def __init__(self):
        super(SingsNet, self).__init__()

        self.n_maps = 32
        self.dropout_rate = 0.5

        # 1st conv gets the input and returns n_maps feat. maps using 3 x 3 filters, stride = 1 and pad = 1
        self.conv1 = nn.Conv2d(3, self.n_maps, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.n_maps)

        # 2nd conv gets previous out and returns n_maps x 2 feat. maps using, 3 x 3 filters, stride =1, pad = 1
        self.conv2 = nn.Conv2d(self.n_maps, self.n_maps * 2, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.n_maps * 2)

        # 3rd conv gets previous out and returns n_maps x 4 feat. maps using, 3 x 3 filters, stride =1, pad = 1
        self.conv3 = nn.Conv2d(self.n_maps * 2, self.n_maps * 4, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(self.n_maps * 4)

        # 2 fully connected layers to transform the output of the convolution layers to the final output
        self.fc1 = nn.Linear(8 * 8 * self.n_maps * 4, self.n_maps * 4)
        self.fcbn1 = nn.BatchNorm1d(self.n_maps * 4)
        self.fc2 = nn.Linear(self.n_maps * 4, 6)

    def forward(self, x):
        x = self.bn1(self.conv1(x))  # batch_size x num_channels x 64 x 64
        x = F.relu(F.max_pool2d(x, 2))  # batch_size x num_channels x 32 x 32
        x = self.bn2(self.conv2(x))  # batch_size x num_channels*2 x 32 x 32
        x = F.relu(F.max_pool2d(x, 2))  # batch_size x num_channels*2 x 16 x 16
        x = self.bn3(self.conv3(x))  # batch_size x num_channels*4 x 16 x 16
        x = F.relu(F.max_pool2d(x, 2))  # batch_size x num_channels*4 x 8 x 8

        # flatten the output for each image
        x = x.view(-1, 8 * 8 * self.n_maps * 4)  # batch_size x 8*8*num_channels*4
        # x = x.view(-1, self.flatten(x))

        # apply 2 fully connected layers with dropout
        x = F.dropout(F.relu(self.fcbn1(self.fc1(x))),
                      p=self.dropout_rate, training=self.training)  # batch_size x self.num_channels*4
        x = self.fc2(x)  # batch_size x 6

        # apply log softmax on each image's output (this is recommended over applying softmax
        # since it is numerically more stable)
        return F.log_softmax(x, dim=1)