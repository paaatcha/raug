#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: André Pacheco
E-mail: pacheco.comp@gmail.com

This file contains functions to load and handle to images in jedy library

If you find any bug or have some suggestion, please, email me.
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from random import shuffle, seed
import glob
import numpy as np
import pandas as pd
import os
import unidecode


def one_hot_encoding(ind, N=None):
    """
    This function binarizes a vector (one hot enconding).
    For example:
    Input: v = [1,2,3]
    Output: v = [[1,0,0;
                0,1,0;
                0,0,1]]

    Parameters:
    ind (array): a array 1 x n
    N (int): the number of indices. If None, the code get is from the shape. Default is None.

    Returns:
    The one hot enconding array n x N

    """

    ind = np.asarray(ind)
    if ind is None:
        return None

    if N is None:
        N = ind.max() + 1

    return (np.arange(N) == ind[:, None]).astype(int)


def get_path_from_folders(path, extra_info_suf=None, img_exts=['png'], shuf=True):
    """
    This function receives a folder path as parameter and returns three lists containing the path folder's children,the
    list of all images inside these folders, and a possible extra information related to each image. If the images have
    extra information, for example, scalar features, you can load them by setting the extra information suffix. If you
    let it as None, the function will return None for the extra information list.

    Note: the extra information must have the same name of the image. You must inform if you included more characters in
    the name using the extra_info_suf. Check the example below:

    For example, supposing the path's folder has the following tree:
    IMGs
        - A
            img1.png
            img1_feat.txt
            img2.png
            img2_feat.txt
        - B
            img3.png
            img3_feat.txt
            img4.png
            img4_feat.txt
    In this case, root folder is IMGs, its children will be A, B. The return lists will be, first, the list of images,
    which each element will be the image path in the format 'IMGs/{children}/img{number}.png'; second the list of images
    extra information in format 'IMGs/{children}/ext_feat1.txt' (recall that you need to set extra_info_suf="_feat.txt"
    to get these features, if you let it None, the list will also be None); third the children folders's ['A', 'B'].

    Parameters:
    path (string): a string with the root folder's path
    extra_info_suf (string): if the images have extra information, and you wanna load them, set the suffix and extension
    here. Default is None
    img_ext (list): a list of images extension to load. Default is only ['png']
    shuf (bool): if you'd like to shuffle the list of images and extra information path. Dafault is True.

    Returns:
    paths (list): a list of images' paths in all folders
    extra_info (list): a list of images' extra information paths in all folders
    fold_names (list): a list of name of all folders' in the root folder
    """

    paths = list()
    fold_names = [nf for nf in os.listdir(path) if os.path.isdir(os.path.join(path, nf))]
    extra_info = list()

    if (len(fold_names) == 0):
        folders = glob.glob(path)
        fold_names = [path.split('/')[-1]]
    else:
        folders = glob.glob(path + '/*')

    for fold in folders:
        for ext in img_exts:
            paths += (glob.glob(fold + '/*.' + ext))

    if (shuf):
        shuffle(paths)

    if (extra_info_suf is not None):
        for p in paths:
            extra_info.append(np.loadtxt(p.split('.')[0] + extra_info_suf, dtype=np.float32))

    return paths, np.asarray(extra_info), fold_names


def load_dataset_from_folders(path, extra_info_suf=None, n_samples=None, img_ext=['png'], shuf=False,
                                     one_hot=True):
    """
    This function receives a folder root path and gets all images, labels and a possible extra information for each image
    in the inner folders. It uses the 'get_path_from_folders' function to load the paths. So, the root folder must be
    organized as described in 'get_path_from_folders'.

    The labels generated is based in the root folder's children. For example, if IMGs is the root folder and we have an
    img in 'IMGs/A/img1.png', the label for all images in folder A will be A.

    Parameters:
    path (string): the root folder's path
    extra_info_suf (string): if the images have extra information, and you wanna load them, set the suffix and extension
    here. Default is None
    n_samples (int): number of samples that you wanna load from the root folder. If None, the function will load all
    images. Default is None.
    img_ext (list): a list of images extension to load. Default is only ['png']
    shuf (bool): if you'd like to shuffle the list of images and extra information path. Dafault is True.
    one_hot (bool): if you'd like the one hot encoding set it as True. Defaul is True.

    Returns:
    img_paths (list): the images' path list containing all images in the root folder's children
    img_labels (list): the labels' list for each image loaded in img_paths
    extra_info (list): the extra information's list for each image loaded in img_paths
    labels_number (dictionary): a python dictionary relating the label and its given number
    """

    img_labels = list()
    labels_number = dict()

    # Getting all paths from 'get_path_from_folders'
    img_paths, extra_info, folds = get_path_from_folders(path, extra_info_suf, img_ext, shuf)

    value = 0
    for f in folds:
        if (f not in labels_number):
            labels_number[f] = value
            value += 1

    if (n_samples is not None):
        img_paths = img_paths[0:n_samples]

    for p in img_paths:
        lab = p.split('/')[-2]
        img_labels.append(labels_number[lab])

    if (one_hot):
        img_labels = one_hot_encoding(img_labels)
    else:
        img_labels = np.asarray(img_labels)

    return img_paths, img_labels, extra_info, labels_number


def load_dataset_from_csv (csv_path, labels_name= None, extra_info_names=None, n_samples=None, img_ext=['png'],
                           shuf=False, seed_number=None, one_hot=True, verbose=True):
    """
    This function loads the dataset considering the data in a .csv file. The .csv structure must be:
    image label, extra information 1, ..., extra information N, and the image path. The extra information is optional,
    if you don't have it, put only the label and the path. The function will always consider the first and the last
    columns as the label and path. Everything between it, will be the extra information, which must be numbers (int or
    floats). However, if you inform the extra_info_names, the function will consider only it. If you let it None, the
    function will consider all of them (if they exist).

    In addition, if labels_name is informed, the function will load only the images and extra information for these list
    of labels. If you let it as None, the function load all labels.

    Parameters:
    csv_path (string): the path to the csv file
    labels_name (list): a list of string containing all labels that must be considered. Default is None.
    extra_info_names (list): list of string containing the extra information name that will be loaded. Default is None.
    n_samples (int): number of samples that you wanna load from the root folder. If None, the function will load all
    images. Default is None.
    img_ext (list): a list of images extension to load. Default is only ['png']
    shuf (bool): if you'd like to shuffle the list of images and extra information path. Dafault is True.
    seed_number (int): the seed number to keep the shuffle for multiples executions. Default is None.
    one_hot (bool): if you'd like the one hot encoding set it as True. Defaul is True.

    Returns:
    img_paths (list): the images' path list containing all images in the root folder's children
    img_labels (list): the labels' list for each image loaded in img_paths
    extra_info (list): the extra information's list for each image loaded in img_paths
    labels_number (dictionary): a python dictionary relating the label and its given number
    verbose (bool): set it as True to print information on the screen. Default is True.
    """

    def format_labels(str_list):
        """
        This is a aux function to format the labels name. It removes some characters and accents, for example.

        Parameter:
        str_list (list): a list of strings or just one string that need to be formatted

        Returns:
        str_list_formatted: a list with all strings formatted
        x: the formatted string
        """

        def format_string_lab(name):
            """
            Function to handle the string and format it
            """
            name = unidecode.unidecode(name).replace(' ', '_').lower()
            str_labs = name.split('_')
            if (len(str_labs) > 1):
                name = '_'.join(str_labs[:-1])
            else:
                name = str_labs[0]
            return name

        if (type(str_list) == list or type(str_list) == np.ndarray):
            for i in range(len(str_list)):
                str_list[i] = format_string_lab(str_list[i])
        else:
            str_list = format_string_lab(str_list)

        return str_list

    # Loading the csv
    csv = pd.read_csv(csv_path)

    # This will be the dataset returned
    dataset = dict()

    # Removing all possible NaN
    csv = csv.dropna()

    # Getting the columns names: the first name must be the label and the last one is the path. If its size is greater
    # than two, it means we also have extra information about the images. If it's true, we set extra_info as True.
    c_names = csv.columns.values
    extra_info = len(c_names) != 2
    label_name = c_names[0]
    path_name = c_names[-1]

    if (extra_info_names is None):
        if (extra_info):
            extra_info_names = c_names[1:-1]

            # Replacing N = 0 e S = 1 for the extra information
            csv[extra_info_names] = csv[extra_info_names].replace(['N', 'S'], [0, 1])

    # Getting the valid labels. In this case, if valid_labes is informed, we need to consider only these labels
    if (labels_name is None):
        labels_name = csv.Diagnostico.unique()

    # Formatting the labels
    labels_name_formatted = format_labels (labels_name)

    if (verbose):
        print ("Loading only the following labels:")
        print (labels_name_formatted)

        if (extra_info):
            print ("\nLoading the following extra information:")
            print (extra_info_names)

        print ("\n")

    # Iterato through all row in the csv
    for k, row in enumerate(csv.iterrows()):
        img_label = (format_labels(row[1][label_name]))
        img_path = row[1][path_name]

        if (img_label not in labels_name_formatted):
            print ("The label {} is not in labels to be selected. I'm skiping it...".format(img_label))
            continue

        if (extra_info):
            extra_info_data = row[1][extra_info_names].tolist();
            dataset[img_path] = (img_label, extra_info_data)
        else:
            dataset[img_path] = img_label

        if (verbose):
            print ('Loading {} - Label: {} - path: {}'.format(k, img_label, img_path))


    if (verbose):
        print ('\n### Data summary: ###\n')
        d = csv.groupby([label_name])[path_name].count()
        print(d)
        print("\n>> Total images: {} <<\n".format(len(dataset)))

    return dataset



