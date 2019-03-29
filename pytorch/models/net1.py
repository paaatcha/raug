#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: André Pacheco
E-mail: pacheco.comp@gmail.com

This file implements a simple convolutional neural network

If you find any bug or have some suggestion, please, email me.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnF


class Net1 (nn.Module):
    """
    Defining the Net1 model following the standard way described in PyTorch tutorials. We need to define the constructor
    and override the forward method. You can have more methods and receive the parameter you need to do so.
    """

    def __init__(self):
        super(Net1, self).__init__()

        self.num_channels = 3

        # Defining the build blocks
        self.conv1 = nn.Conv2d(self.num_channels, )



