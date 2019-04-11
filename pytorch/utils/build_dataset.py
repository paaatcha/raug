#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: André Pacheco
E-mail: pacheco.comp@gmail.com

This file implements the methods and functions to load the image as a PyTorch dataset

If you find any bug or have some suggestion, please, email me.
"""

from PIL import Image
from torch.utils import data
import torchvision.transforms as transforms


class BuildDataset (data.Dataset):
    """
    This the standard way to implement a dataset pipeline in PyTorch. We need to extend the torch.utils.data.Dataset
    class and implement the following methods: __len__, __getitem__ and the constructor __init__
    """

    def __init__(self, imgs_path, labels, extra_info=None, transform=None):
        """
        The constructor gets the images path and their respectively labels and extra information (if it exists).
        In addition, you can specify some transform operation to be carry out on the images.

        It's important to note the images must match with the labels (an extra information if exist). For example, the
        imgs_path[x]'s label must take place on labels[x].

        Parameters:
        :param imgs_path (list): a list of string containing the image paths
        :param labels (list) a list of labels for each image
        :param extra_info (list): a list of extra information regarding each image. If None, there is no information.
        Defaul is None.
        :param transform (torchvision.transforms.transforms.Compose): transform operations to be carry out on the images
        """

        self.imgs_path = imgs_path
        self.labels = labels
        self.extra_info = extra_info

        # if transform is None, we need to ensure that the PIL image will be transformed to tensor, otherwise we'll got
        # an exception
        if (transform is not None):
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor()
            ])

    def __len__(self):
        """ This method just returns the dataset size """
        return len(self.imgs_path)

    def __getitem__(self, item):
        """
        It gets the image, labels and extra information (if it exists) according to the index informed in `item`.
        It also performs the transform on the image.

        :param item (int): an index in the interval [0, ..., len(img_paths)-1]
        :return (tuple): a tuple containing the image, its label and extra information (if it exists)
        """

        image = Image.open(self.imgs_path[item]).convert("RGB")

        # Applying the transformations
        image = self.transform(image)

        # print(self.imgs_path[item])
        # print(self.labels[item])
        # print(self.extra_info[item])

        if (self.extra_info is None):
            return image, self.labels[item], []
        else:
            return image, self.labels[item], self.extra_info[item]


def get_data_loader (imgs_path, labels, extra_info=None, transform=None, params=None):
    """
    This function gets a list og images path, their labels and extra information (if it exists) and returns a DataLoader
    for these files. You also can set some transformations using torchvision.transforms in order to perform data
    augmentation. Lastly, params is a dictionary that you can set the following parameters:
    batch_size (int): the batch size for the dataset. If it's not informed the default is 30
    shuf (bool): set it true if wanna shuffe the dataset. If it's not informed the default is True
    num_workers (int): the number thread in CPU to load the dataset. If it's not informed the default is 0 (which


    :param imgs_path (list): a list of string containing the images path
    :param labels (list): a list of labels for each image
    :param extra_info (list, optional): a list of extra information regarding each image. If it's None, it means there's
    no extra information. Default is None
    :param transform (torchvision.transforms, optional): use the torchvision.transforms.compose to perform the data
    augmentation for the dataset. Alternatively, you can use the jedy.pytorch.utils.augmentation to perform the
    augmentation. If it's None, none augmentation will be perform. Default is None
    :param params (dictionary, optional): this dictionary contains the following parameters:
    batch_size: the batch size. If the key is not informed or params = None, the default value will be 30
    shuf: if you'd like to shuffle the dataset. If the key is not informed or params = None,
           the default value will be True
    num_workers: the number of threads to be used in CPU. If the key is not informed or params = None, the default
                 value will be  4
    pin_memory = set it to True to Pytorch preload the images on GPU. If the key is not informed or params = None,
                 the default value will be True
    :return (torch.utils.data.DataLoader): a dataloader with the dataset and the chose params
    """


    dt = BuildDataset(imgs_path, labels, extra_info, transform)

    # Checking the params values. If it's not defined in params of if params is None, the default values are described
    # below:
    batch_size = 30
    shuf = True
    num_workers = 4
    pin_memory = True

    # However, if the params is defined, we used the values described on it:
    if (params is not None):
        if ('batch_size' in params.keys()):
            batch_size = params['batch_size']
        if ('shuf' in params.keys()):
            shuf = params['shuf']
        if ('num_workers' in params.keys()):
            num_workers = params['num_workers']
        if ('pin_memory' in params.keys()):
            pin_memory = params['pin_memory']

    # Calling the dataloader
    dl = data.DataLoader (dataset=dt, batch_size=batch_size, shuffle=shuf, num_workers=num_workers,
                          pin_memory=pin_memory)

    return dl
