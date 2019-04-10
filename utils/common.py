#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: André Pacheco
E-mail: pacheco.comp@gmail.com

This file contains common and auxiliary functions to be used in the package

If you find any bug or have some suggestion, please, email me.
"""

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms.functional as trans
from torchvision.utils import make_grid
import PIL


def flatten(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def one_hot_encoding(ind, N=None):
    """
    This function binarizes a vector (one hot enconding).
    For example:
    Input: v = [1,2,3]
    Output: v = [[1,0,0;
                0,1,0;
                0,0,1]]

    :param ind (numpy array): an numpy array 1 x n in which each position is a label
    :param N (int, optional): the number of indices. If None, the code get is from the shape. Default is None.

    :return (numpy.array): the one hot enconding array n x N
    """

    ind = np.asarray(ind)
    if ind is None:
        return None

    if N is None:
        N = ind.max() + 1

    return (np.arange(N) == ind[:, None]).astype(int)


def create_folders(path, folders, train_test_val=False):
    """
    This function creates a folder tree inside a root folder's path informed as parameter.

    :param path (string): the root folder's path
    :param folders (list): a list of strings representing the name of the folders will be created inside the root.
    :param train_test_val (bool, optional): if you wanns create TRAIN, TEST and VAL partition for each folder.
    Default is False.
    """

    # Checking if the folder doesn't exist. If True, we must create it.
    if (not os.path.isdir(path)):
        os.mkdir(path)

    if (train_test_val):
        if (not os.path.isdir(os.path.join(path,'TEST'))):
            os.mkdir(os.path.join(path, 'TEST'))
        if (not os.path.isdir(os.path.join(path, 'TRAIN'))):
            os.mkdir(os.path.join(path, 'TRAIN'))
        if (not os.path.isdir(os.path.join(path, 'VAL'))):
            os.mkdir(os.path.join(path, 'VAL'))

    for folder in folders:
        if (train_test_val):
            if (not os.path.isdir(os.path.join(path, 'TRAIN', folder))):
                os.mkdir(os.path.join(path, 'TRAIN', folder))
            if (not os.path.isdir(os.path.join(path, 'TEST', folder))):
                os.mkdir(os.path.join(path, 'TEST', folder))
            if (not os.path.isdir(os.path.join(path, 'VAL', folder))):
                os.mkdir(os.path.join(path, 'VAL', folder))
        else:
            if (not os.path.isdir(os.path.join(path, folder))):
                os.mkdir(os.path.join(path, folder))


def create_folders_from_iterator (folder_path, iter):
    """
    Function to create a list of paths gitven a iterator
    :param folder_path (string): the root path
    :param iter (iterator): a list, tuple or dict to create the folders
    :return:
    """

    # Getting the dict keys as a list
    if (type(iter) is dict):
        iter = iter.keys()

    # Checking if the folder doesn't exist. If True, we must create it.
    if (not os.path.isdir(folder_path)):
        os.mkdir(folder_path)

    # creating the folders
    for i in iter:
        if (not os.path.isdir(os.path.join(folder_path, i))):
            os.mkdir(os.path.join(folder_path, i))



def convert_colorspace(img_path, colorspace):
    """
    This function receives an RGB image path, load it and convert it to the desired colorspace

    :param img_path (string): the RGB image path
    :param colorspace (string): the following colorpace: ('HSV', 'Lab', 'XYZ', 'HSL' or 'YUV')

    :return (numpy.array):
    img: the converted image
    """

    img = cv2.imread(img_path)

    # In this case, the path is wrong
    if (img is None):
        raise FileNotFoundError

    if (colorspace == 'HSV'):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif (colorspace == 'Lab'):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    elif (colorspace == 'XYZ'):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)
    elif (colorspace == 'HLS'):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    elif (colorspace == 'YUV'):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    else:
        raise Exception ("There is no conversion for {}".format(colorspace))

    return img


def plot_img_data_loader (data_loader, n_img_row=4):

    # Getting some samples to plot the images
    data_sample = iter(data_loader)
    data = data_sample.next()
    if (len(data) == 2):
        images, labels = data
    else:
        images, labels = data[0:2]

    # show images
    plot_img (make_grid(images, nrow=n_img_row, padding=10), True)

    # print labels
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


def plot_img (img, grid=False, title="Image test", hit=False, save_path=None):

    if (grid):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    else:
        if (type(img) == PIL.Image.Image):
            plt.imshow(np.array(img))
        else:
            npimg = img.numpy()
            img = np.moveaxis(npimg[:, :, :], 0, -1)
            plt.imshow(img)

    if (hit):
        title_obj = plt.title(title)
        plt.setp(title_obj, color='g')
    else:
        title_obj = plt.title(title)
        plt.setp(title_obj, color='r')

    if (save_path is None):
        plt.show()
    else:
        plt.savefig(save_path)

