#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: André Pacheco
E-mail: pacheco.comp@gmail.com

This file implements different networks that you can use for your task.
Note that you don't need to use them. You can build your own model using nn.Sequential, for example.

If you find any bug or have some suggestion, please, email me.
"""

import torch.nn as nn
import torch.nn.functional as F
import torch


class PadNet(nn.Module):

    # Vai entrar um imagem (N, 3, 224, 224)
    def __init__(self):
        super(PadNet, self).__init__()

        self.n_maps = 16
        self.dropout_rate = 0.5


        self.conv1 = nn.Conv2d(3, self.n_maps, 5, stride=1)
        self.bn1 = nn.BatchNorm2d(self.n_maps)

        self.conv2 = nn.Conv2d(self.n_maps, self.n_maps * 2, 5, stride=1)
        self.bn2 = nn.BatchNorm2d(self.n_maps * 2)

        self.conv3 = nn.Conv2d(self.n_maps * 2, self.n_maps * 4, 3, stride=1)
        self.bn3 = nn.BatchNorm2d(self.n_maps * 4)

        self.conv4 = nn.Conv2d(self.n_maps * 4, self.n_maps * 5, 3, stride=1)
        self.bn4 = nn.BatchNorm2d(self.n_maps * 5)

        self.conv5 = nn.Conv2d(self.n_maps * 5, self.n_maps * 5, 3, stride=1)
        self.bn5 = nn.BatchNorm2d(self.n_maps * 5)

        self.fc1 = nn.Linear(4 * 4 * self.n_maps * 5, self.n_maps * 10)
        self.fcbn1 = nn.BatchNorm1d(self.n_maps * 10)

        self.fc2 = nn.Linear(self.n_maps * 10, self.n_maps * 6)
        self.fcbn2 = nn.BatchNorm1d(self.n_maps * 6)

        self.fc3 = nn.Linear(self.n_maps * 6, 6)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)


    def forward(self, x):

        # 224 x 224
        x = self.bn1(self.conv1(x))
        x = self.relu(self.maxpool(x))

        x = self.bn2(self.conv2(x))
        x = self.relu(self.maxpool(x))

        x = self.bn3(self.conv3(x))
        x = self.relu(self.maxpool(x))

        x = self.bn4(self.conv4(x))
        x = self.relu(self.maxpool(x))

        x = self.bn5(self.conv5(x))
        x = self.relu(self.maxpool(x))

        # flatten the output for each image
        x = x.view(-1, 4 * 4 * self.n_maps * 5)

        x = F.dropout(F.relu(self.fcbn1(self.fc1(x))),
                      p=self.dropout_rate, training=self.training)

        x = F.dropout(F.relu(self.fcbn2(self.fc2(x))),
                      p=self.dropout_rate, training=self.training)

        x = self.fc3(x)  # batch_size x 6

        return F.log_softmax(x, dim=1)


class PadNetFeat(nn.Module):

    # Vai entrar um imagem (N, 3, 224, 224)
    def __init__(self):
        super(PadNetFeat, self).__init__()

        self.n_maps = 16
        self.dropout_rate = 0.5


        self.conv1 = nn.Conv2d(3, self.n_maps, 5, stride=1)
        self.bn1 = nn.BatchNorm2d(self.n_maps)

        self.conv2 = nn.Conv2d(self.n_maps, self.n_maps * 2, 5, stride=1)
        self.bn2 = nn.BatchNorm2d(self.n_maps * 2)

        self.conv3 = nn.Conv2d(self.n_maps * 2, self.n_maps * 4, 3, stride=1)
        self.bn3 = nn.BatchNorm2d(self.n_maps * 4)

        self.conv4 = nn.Conv2d(self.n_maps * 4, self.n_maps * 5, 3, stride=1)
        self.bn4 = nn.BatchNorm2d(self.n_maps * 5)

        self.conv5 = nn.Conv2d(self.n_maps * 5, self.n_maps * 5, 3, stride=1)
        self.bn5 = nn.BatchNorm2d(self.n_maps * 5)

        self.fc1 = nn.Linear(4 * 4 * self.n_maps * 5, self.n_maps * 10)
        self.fcbn1 = nn.BatchNorm1d(self.n_maps * 10)

        self.fc2 = nn.Linear(self.n_maps * 10, self.n_maps * 6)
        self.fcbn2 = nn.BatchNorm1d(self.n_maps * 6)

        self.fc3 = nn.Linear((self.n_maps * 6) + 24, 6)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, img, feat):

        # 224 x 224
        img = self.bn1(self.conv1(img))
        img = self.relu(self.maxpool(img))

        img = self.bn2(self.conv2(img))
        img = self.relu(self.maxpool(img))

        img = self.bn3(self.conv3(img))
        img = self.relu(self.maxpool(img))

        img = self.bn4(self.conv4(img))
        img = self.relu(self.maxpool(img))

        img = self.bn5(self.conv5(img))
        img = self.relu(self.maxpool(img))

        # flatten the output for each image
        img = img.view(-1, 4 * 4 * self.n_maps * 5)

        img = F.dropout(F.relu(self.fcbn1(self.fc1(img))),
                        p=self.dropout_rate, training=self.training)

        img = F.dropout(F.relu(self.fcbn2(self.fc2(img))),
                        p=self.dropout_rate, training=self.training)

        agg = torch.cat((img, feat), dim=1)

        res = self.fc3(agg)

        return F.log_softmax(res, dim=1)


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


def _flatten(x):
    """
    Auxiliary function to get the flat features

    :param x (torch.Tensor): a tensor with the shape (batch_size, channels, width, height)
    :return (int): the number of feature to be carried out to the next layer
    """
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features
