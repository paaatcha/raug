#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: André Pacheco
E-mail: pacheco.comp@gmail.com

This file implements the Context Guided Cell (GCell)

If you find any bug or have some suggestion, please, email me.
"""

import torch.nn as nn
import torch

class MetaNet(nn.Module):
    """
    Implementing the MetaNet approach
    Fusing Metadata and Dermoscopy Images for Skin Disease Diagnosis - https://ieeexplore.ieee.org/document/9098645
    """
    def __init__(self, in_channels, middle_channels, out_channels):
        super(MetaNet, self).__init__()
        self.metanet = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 1),
            nn.ReLU(),
            nn.Conv2d(middle_channels, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, feat_maps, metadata):
        metadata = torch.unsqueeze(metadata, -1)
        metadata = torch.unsqueeze(metadata, -1)
        x = self.metanet(metadata)
        x = x * feat_maps
        return x
