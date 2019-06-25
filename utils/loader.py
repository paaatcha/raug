#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: André Pacheco
E-mail: pacheco.comp@gmail.com

This file contains functions and auxiliary functions to load and handle images in jedy package

If you find any bug or have some suggestion, please, email me.
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from random import shuffle, seed
from sklearn.model_selection import KFold
import glob
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
from collections import Counter
import unidecode
import cv2
import shutil
from .common import one_hot_encoding, create_folders, convert_colorspace



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

    :param path (string): a string with the root folder's path
    :param extra_info_suf (string, optional): if the images have extra information, and you wanna load them, set the
     suffix and extension here. Default is None
    :param img_ext (list, optional): a list of images extension to load. Default is only ['png']
    :param shuf (bool, optional ): if you'd like to shuffle the list of images and extra information path.
    Default is True.

    :return (tuple): a tuple containing:
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
        folders = glob.glob(os.path.join(path, '*'))

    for fold in folders:
        for ext in img_exts:
            paths += (glob.glob(os.path.join(fold, '*.' + ext)))

    if (len(paths) == 0):
        raise Exception ("There is no image with the extensions {} in the given path".format(img_exts))

    if (shuf):
        shuffle(paths)

    if (extra_info_suf is not None):
        for p in paths:
            extra_info.append(np.loadtxt(p.split('.')[0] + extra_info_suf, dtype=np.float32))

        return paths, np.asarray(extra_info), fold_names
    else:
        return paths, None, fold_names


def load_dataset_from_folders(path, extra_info_suf=None, n_samples=None, img_ext=['png'], shuf=False, one_hot=True,
                              label_str=False):
    """
    This function receives a folder root path and gets all images, labels and a possible extra information for each image
    in the inner folders. It uses the 'get_path_from_folders' function to load the paths. So, the root folder must be
    organized as described in 'get_path_from_folders'.

    The labels generated is based in the root folder's children. For example, if IMGs is the root folder and we have an
    img in 'IMGs/A/img1.png', the label for all images in folder A will be A.

    :param path (string): the root folder's path
    :param extra_info_suf (string, optional): if the images have extra information, and you wanna load them, set the
    suffix and extension here. Default is None
    :param n_samples (int, optional): number of samples that you wanna load from the root folder. If None, the function
    will load all images. Default is None.
    :param img_ext (list, optional): a list of images extension to load. Default is only ['png']
    :param shuf (bool, optional): if you'd like to shuffle the list of images and extra information path.
    Default is True.
    :param one_hot (bool, optional): if you'd like the one hot encoding set it as True. Default is True.
    :param label_str (bool, optional): if you'd like to load the labels as string or number. For example, you can have
    the labels like ['A', 'B'] or [0, 1], in which 0 means A and 1 means B. If you set it as True, labels_number will
    return None and one_hot has no effect. Default is False.

    :return (tuple): a tuple containing:
    img_paths (list): the images' path list containing all images in the root folder's children
    img_labels (list): the labels' list for each image loaded in img_paths
    extra_info (list): the extra information's list for each image loaded in img_paths
    labels_number (dictionary): a python dictionary relating the label and its given number
    """

    img_labels = list()
    labels_number = dict()

    # Getting all paths from 'get_path_from_folders'
    img_paths, extra_info, folds = get_path_from_folders(path, extra_info_suf, img_ext, shuf)

    if (not label_str):
        # Sorting the folders to get the labels numbers in alphabetic order
        folds.sort()

        value = 0
        for f in folds:
            if (f not in labels_number):
                labels_number[f] = value
                value += 1
    else:
        labels_number = None

    if (n_samples is not None):
        img_paths = img_paths[0:n_samples]

    for p in img_paths:
        lab = p.split('/')[-2]
        if (not label_str):
            img_labels.append(labels_number[lab])
        else:
            img_labels.append(lab)

    if (not label_str):
        if (one_hot):
            img_labels = one_hot_encoding(img_labels)
        else:
            img_labels = np.asarray(img_labels)

    # Getting the frequency for each label. It's very useful to weight the dataset
    labels_count = dict(Counter(img_labels))

    return img_paths, img_labels, extra_info, labels_number, labels_count


def load_dataset_from_csv (csv_path, labels_name=None, extra_info_names=None, extra_info_str=None,
                           drop_na=True, verbose=True, include_ext=None, str_label=False):
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
    :param csv_path (string): the path to the csv file
    :param labels_name (list, optional): a list of string containing all labels that must be considered. If None,
    it considers all of them. Default is None.
    :param extra_info_names (list, optional): list of string containing the extra information name that will be loaded.
    If None, it considers all of them. Default is None.
    :param extra_info_str (string, optional): if you have any extra info that is stored as string and you wanna
    transform it to one hot enconding, type its name here. Default is None.
    :param drop_na (bool, optional): if you set it True any missing data from the CSV will be removed. Default is True.
    :param verbose (bool, optional): if you'd like to print some information on the screen set it True. Default is False
    :param include_ext (string, optional): if you'd like to include a extension into your image set the extension here.
    If it's None, you're assuming all images already have the extension in the path. Default is None.
    :param str_label (bool, optional) if you'd like to let the labels for each image as string, set it True, otherwise
    it's gonna be numbers. Default is False.

    :return a dict with the image name as key and extra info and label as values
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

    if drop_na:
        # Removing all possible NaN
        csv = csv.dropna()
    else:
        csv = csv.fillna(-1)

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

            # Replacing Female = 0 e Male = 1 for the extra information
            csv[extra_info_names] = csv[extra_info_names].replace(['female', 'male'], [0, 1])

    # In this case, we need to convert the string to one hot encode
    extra_info_str2num = dict()
    if (extra_info_str is not None):
        if (extra_info):
            extra_info_str_values = csv[extra_info_str].unique()
            size_v = len(extra_info_str_values)
            num_one_hot = one_hot_encoding(list(range(size_v)))

            for num, val in enumerate(extra_info_str_values):
                extra_info_str2num[val] = num_one_hot[num]

        # Fixing the value when it is missing
        if (-1 in extra_info_str2num.keys()):
            extra_info_str2num[-1] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

    # Getting the valid labels. In this case, if valid_labes is informed, we need to consider only these labels
    if (labels_name is None):
        labels_name = csv.diagnostic.unique()

    # Formatting the labels
    labels_name_formatted = format_labels (labels_name)
    labels_name_formatted.sort() # just to get the labels in alphabetic order


    if (verbose):
        print ("Loading only the following labels:")
        print (labels_name_formatted)

        if (extra_info):
            print ("\nLoading the following extra information:")
            print (extra_info_names)

        print ("\n")

    # This is used to set the labels to number instead of strings
    if not str_label:
        label_str2number_dict = {key : val for val, key in enumerate(labels_name_formatted)}

    # Setting tqdm to show some information on the screen
    with tqdm(total=len(csv), ascii=True, ncols=100) as t:

        # Iterating through all row in the csv
        for k, row in enumerate(csv.iterrows()):

            if str_label:
                img_label = (format_labels(row[1][label_name]))
                if (img_label not in labels_name_formatted):
                    print("The label {} is not in labels to be selected. I'm skiping it...".format(img_label))
                    continue
            else:
                try:
                    img_label = label_str2number_dict[format_labels(row[1][label_name])]
                except:
                    print("The label {} is not in labels to be selected. I'm skiping it...".format(img_label))
                    continue

            if include_ext is None:
                img_path = row[1][path_name]
            else:
                img_path = row[1][path_name] + include_ext

            # making some convertions to put all regions in one hot enconding
            if (extra_info):

                if (extra_info_str is not None):
                    v = row[1][extra_info_str]
                    row[1][extra_info_str] = extra_info_str2num[v]

                    # print(row[1][extra_info_str])

                info = row[1][extra_info_names].tolist();

                extra_info_data = []
                for sub in info:

                    if (type(sub) is np.ndarray):
                        for item in sub:
                             extra_info_data.append(item)
                    else:
                        extra_info_data.append(sub)

                dataset[img_path] = (img_label, extra_info_data)
            else:
                dataset[img_path] = img_label

            # Updating tqdm
            t.update()

    pd_summary = csv.groupby([label_name])[path_name].count()
    labels_name = list(pd_summary.index)
    labels_freq = pd_summary.values
    if (verbose):
        print ('\n### Data summary: ###\n')
        print(pd_summary)
        print("\n>> Total images: {} <<\n".format(len(dataset)))

    return dataset, labels_name, extra_info_str2num, labels_freq


def _get_lists_from_dict (keys, dataset, base_path, extra_info):
    """
    Auxiliary function for dataset_k_folder_from_dict and split_dataset_from_dict
    """
    imgs_path_list = list()
    labels_list = list()
    if extra_info:
        extra_info_list = list()
    else:
        extra_info_list = None
    for key in keys:
        if base_path is not None:
            imgs_path_list.append(os.path.join(base_path, key))
        else:
            imgs_path_list.append(key)
        if extra_info:
            labels_list.append(dataset[key][0])
            extra_info_list.append(dataset[key][1])
        else:
            labels_list.append(dataset[key][0])

    return imgs_path_list, labels_list, extra_info_list

def dataset_k_folder_from_dict (dataset, base_path=None, k=5, extra_info=False, tr=0.85, te=0.15,
                                   seed_number=None):
    """
    This function returns the k folder dataset in order to perform cross validation.
    :param dataset (dict): a dictionary obtained by the CSV data. 1st perform the load_dataset_from_csv in order to
    get this dict().
    :param base_path (string, optional): the base path to the images. Default is None.
    :param k (number, optional): the number of folders. Default is 5.
    :param extra_info (bool, optional): if you have extra information set it as True. Default is False.
    :param tr (number, optional): the % of data to share between train and val partitions. Default is 0.85
    :param te (number, optional): the % of data to use in the test partition
    :param seed_number (number, optional): a seed number to guarantee reproducibility
    :return (tuple):
    Position 0: a dict which the keys as the "folder_1" to "folder_k". Each value for each key will contain a list like:
    [data train, data_val]. Either data_train or data_val contain 3 lists like: [imgs_path, labels, extra_info]

    Position 1: a list containing the test data like: [imgs_path, labels, extra_info]
    """

    print ("Generating the {}-folders...".format(k))
    # Checking the % for the partitions
    if abs(1.0 - tr - te) >= 0.01:
        raise Exception('The values of tr and te must sum up 1.0')

    # Setting the seed to reproduce the results later
    if seed_number is not None:
        seed(seed_number)

    all_keys = list(dataset.keys())
    shuffle(all_keys)

    # Splitting the partitions
    N = len(all_keys)
    n_test = int(round(te * N))
    n_train_val = N - n_test

    all_test_keys = all_keys[0:n_test]
    all_train_keys = all_keys[n_test:(n_test+n_train_val)]

    # Generating the test folder
    test_folder = _get_lists_from_dict(all_test_keys, dataset, base_path, extra_info)

    # Now we need to generate F folders using the all_train_keys
    all_train_keys = np.array(all_train_keys)
    dict_folders = dict()
    kfold = KFold (k, True, seed_number)
    j = 0

    for train_idx, val_idx in kfold.split(all_train_keys):
        train_keys = all_train_keys[train_idx]
        val_keys = all_train_keys[val_idx]

        train_folder = _get_lists_from_dict(train_keys, dataset, base_path, extra_info)
        val_folder = _get_lists_from_dict(val_keys, dataset, base_path, extra_info)

        fd_str = 'folder_{}'.format(j)
        dict_folders[fd_str] = (train_folder, val_folder)
        j+=1

    return dict_folders, test_folder


def split_dataset_from_dict (dataset, base_path=None, extra_info=False, tr=0.80, tv= 0.10, te=0.10, seed_number=None):
    """
    This function returns the dataset slitted in 3 partitions: train, validation and test
    :param dataset (dict): a dictionary obtained by the CSV data. 1st perform the load_dataset_from_csv in order to
    get this dict().
    :param base_path (string, optional): the base path to the images. Default is None.
    :param extra_info (bool, optional): if you have extra information set it as True. Default is False.
    :param tr (number, optional): the % of data to share between train and val partitions. Default is 0.8
    :param te (number, optional): the % of data to use in the test partition. Default is 0.10
    :param tv (number, optional): the % of data to use in the validation partition. Default is 0.10
    :param seed_number (number, optional): a seed number to guarantee reproducibility
    :return (tuple):
    Position 0: the train partition containing a list like: [imgs_path, labels, extra_info]
    Position 1: the validation partition containing a list like: [imgs_path, labels, extra_info]
    position 2: the test partition containing a list like: [imgs_path, labels, extra_info]
    """
    # Checking the % for the partitions
    if abs(1.0 - tr - te - tv) >= 0.01:
        raise Exception('The values of tr and te must sum up 1.0')

    # Setting the seed to reproduce the results later
    if seed_number is not None:
        seed(seed_number)

    all_keys = list(dataset.keys())
    shuffle(all_keys)

    # Splitting the partitions
    N = len(all_keys)
    n_test = int(round(te * N))
    n_val = int(round(tv * N))
    n_train = N - n_test - n_val

    all_test_keys = all_keys[0:n_test]
    all_val_keys = all_keys[n_test:(n_test + n_val)]
    all_train_keys = all_keys[(n_test + n_val):(n_test + n_val + n_train)]

    train_data = _get_lists_from_dict(all_train_keys, dataset, base_path, extra_info)
    val_data = _get_lists_from_dict(all_val_keys, dataset, base_path, extra_info)
    test_data = _get_lists_from_dict(all_test_keys, dataset, base_path, extra_info)

    return train_data, val_data, test_data


def dataset_k_folder (imgs_path, labels, extra_info=None, k=5, tr=0.85, te=0.15, seed_number=None):
    """
    This function returns the k folder dataset in order to perform cross validation.
    :param imgs_path (list): a list of string with all images path
    :param labels (list): a list of labels for each image
    :param extra_info (list, optional): a list with the extra information regarding the imgs_path. If it's None, there's
    no extra information. Default is None.
    :param k (number, optional): the number of folders. Default is 5.
    :param extra_info (bool, optional): if you have extra information set it as True. Default is False.
    :param tr (number, optional): the % of data to share between train and val partitions. Default is 0.85
    :param te (number, optional): the % of data to use in the test partition
    :param seed_number (number, optional): a seed number to guarantee reproducibility
    :return (tuple):
    Position 0: a dict which the keys as the "folder_1" to "folder_k". Each value for each key will contain a list like:
    [data train, data_val]. Either data_train or data_val contain 3 lists like: [imgs_path, labels, extra_info]

    Position 1: a list containing the test data like: [imgs_path, labels, extra_info]
    """

    print("Generating the {}-folders...".format(k))

    # Checking the % for the partitions
    if abs(1.0 - tr - te) >= 0.01:
        raise Exception('The values of tr and te must sum up 1.0')


    # Splitting the partitions
    N = len(imgs_path)
    n_test = int(round(te * N))
    n_train = N - n_test

    imgs_path_test = imgs_path[0:n_test]
    imgs_path_train = imgs_path[n_test:(n_test + n_train)]

    labels_test = labels[0:n_test]
    labels_train = labels[n_test:(n_test + n_train)]

    if (extra_info is not None):
        extra_info_test = extra_info[0:n_test]
        extra_info_train = extra_info[n_test:(n_test + n_train)]
    else:
        extra_info_test = None
        extra_info_train = None

    # Generating the test folder
    test_folder = (imgs_path_test, labels_test, extra_info_test)

    # Now we need to generate F folders using the all_train_keys
    imgs_path_train = np.array(imgs_path_train)
    labels_train = np.array(labels_train)
    if extra_info_train is not None:
        extra_info_train = np.array(extra_info_train)
    dict_folders = dict()
    kfold = KFold (k, True, seed_number)
    j = 0

    for train_idx, val_idx in kfold.split(imgs_path_train):

        if extra_info is not None:
            train_folder = (imgs_path_train[train_idx], labels_train[train_idx], extra_info_train[train_idx])
            val_folder = (imgs_path_train[val_idx], labels_train[val_idx], extra_info_train[val_idx])
        else:
            train_folder = (imgs_path_train[train_idx], labels_train[train_idx], None)
            val_folder = (imgs_path_train[val_idx], labels_train[val_idx], None)

        fd_str = 'folder_{}'.format(j)
        dict_folders[fd_str] = (train_folder, val_folder)
        j+=1

    return dict_folders, test_folder



def split_dataset (imgs_path, labels, extra_info=None, sets_perc=[0.8, 0.1, 0.1], verbose=True):
    """
    This function receives 2 or 3 lists (depends on the extra information) and split them into the train, validation and
    test partitions.

    It's important to note the images must match with the labels (an extra information if exist). For example, the
    imgs_path[x]'s label must take place on labels[x].

    :param imgs_path (list): a list of string with all images path
    :param labels (list): a list of labels for each image
    :param extra_info (list, optional): a list with the extra information regarding the imgs_path. If it's None, there's
    no extra information. Default is None.
    :param sets_perc (list, optional): a list with the % values for train, validation and test. It must sum up 1. Default is [0.8,
    0.1, 0.1]
    :param verbose (boolean): wether you'd like to print information on the screen. Default is True.
    :return: 2 or 3 tuples (depends on the extra information) with the partitions values
    """

    # Checking if all sets have the same size:
    if (len(imgs_path) != len(labels)):
        raise Exception ("The length of imgs_path is different than labels. It's not allowed")
    if (extra_info is not None):
        raise Exception ("The length of imgs_path is different than extra_info. It's not allowed")


    # Checking the % for the partitions
    if (abs(1.0 - sum(sets_perc)) >= 0.01):
        print('The % in sets_perc must sum up 1.0')
        raise ValueError

    # Splitting the partitions
    _, tv, te = sets_perc
    N = len(imgs_path)
    n_test = int(round(te * N))
    n_val = int(round(tv * N))
    n_train = N - n_test - n_val

    if (verbose):
        print('Summary:')
        print('Total: ', N, ' | Train: ', n_train, ' | Test: ', n_test, ' | Val: ', n_val, '\n')

    imgs_path_test = imgs_path[0:n_test]
    imgs_path_val = imgs_path[n_test:(n_test + n_val)]
    imgs_path_train = imgs_path[(n_test + n_val):(n_test + n_val + n_train)]

    labels_test = labels[0:n_test]
    labels_val = labels[n_test:(n_test + n_val)]
    labels_train = labels[(n_test + n_val):(n_test + n_val + n_train)]

    if (extra_info is not None):
        extra_info_test = extra_info[0:n_test]
        extra_info_val = extra_info[n_test:(n_test + n_val)]
        extra_info_train = extra_info[(n_test + n_val):(n_test + n_val + n_train)]

        return (imgs_path_train, imgs_path_val, imgs_path_test), (labels_train, labels_val, labels_test), \
               (extra_info_train, extra_info_val, extra_info_test)

    else:
        return (imgs_path_train, imgs_path_val, imgs_path_test), (labels_train, labels_val, labels_test), (None, None, None)


def split_folders_train_test_val(path_in, path_out, extra_info_suf=None, img_ext=["png"], sets_perc=[0.8, 0.1, 0.1],
                                 shuf=True, seed_number=None, verbose=False):
    """
    This function gets a root folder path in path_in without the train, validation and test partition and creates, in
    path_out a dataset considering all partitions. All data inside path_in is copied to path_out. So, you don't lose
    your original data. It's easier to understand using an example:

    Let us consider you have the following Dataset in path_in:
    Dataset:
        A:
            img...
        B:
            img...

    The function will return a new dataset in path_out according to the following structure:
    Dataset:
        TRAIN:
            A:
                imgs...
            B:
                imgs...
        TEST:
            A:
                imgs...
            B:
                imgs...
        VAL:
            A:
                imgs...
            B:
                imgs...

    :param path_in (string): the root folder path that you wanna split in the train, test and val partitions
    :param path_out (string): the root folder path that will receive the new folder structure
    :param extra_info_suf (string, optional): if the images have an extra information file you must inform the suffix
    (see get_path_from_folders in loader.py for more information). Default is None.
    :param sets_perc (list, optional): the set percentage of data for [train, val, test]. It must sum up 1.0.
    Default is [0.8, 0.1, 0.1]
    :param shuf (bool, optional): set it as True if you wanna shuffle the images. Default is True.
    :param seed_number (int, optional): the seed number to keep the shuffle for multiples executions. Default is None.
    :param verbose (bool, optional): set it as True to print information on the screen. Default is True.

    """

    # Checking the % for the partitions
    if (abs(1.0-sum(sets_perc)) >= 0.01):
        raise Exception ('The % in sets_perc must sum up 1.0')

    # Getting all folders in the root folder path
    folders = [nf for nf in os.listdir(path_in) if os.path.isdir(os.path.join(path_in, nf))]

    # Calling create folders to create the new folder structure considering Train, Val and Test partitions
    create_folders(path_out, folders, True)

    for lab in folders:
        for ext in img_ext:
            path_imgs = glob.glob(os.path.join(path_in, lab, "*." + ext))

        if shuf:
            # This is used to keep the same partitions for each train, val and test sets
            if (seed_number is not None):
                seed(seed_number)
            shuffle(path_imgs)

        # Splitting the partitions
        _, tv, te = sets_perc
        N = len(path_imgs)
        n_test = int(round(te * N))
        n_val = int(round(tv * N))
        n_train = N - n_test - n_val

        if (verbose):
            print('Working on ', lab)
            print('Total: ', N, ' | Train: ', n_train, ' | Test: ', n_test, ' | Val: ', n_val, '\n')

        path_test = path_imgs[0:n_test]
        path_val = path_imgs[n_test:(n_test + n_val)]
        path_train = path_imgs[(n_test + n_val):(n_test + n_val + n_train)]

        if (extra_info_suf is None):
            for p in path_test:
                shutil.copy(p, os.path.join(path_out, 'TEST', lab))

            for p in path_train:
                shutil.copy(p, os.path.join(path_out, 'TRAIN', lab))

            for p in path_val:
                shutil.copy(p, os.path.join(path_out, 'VAL', lab))

        else:
            for p in path_test:
                shutil.copy(p, os.path.join(path_out, 'TEST', lab))
                shutil.copy(p.split('.')[0] + extra_info_suf, os.path.join(path_out, 'TEST', lab))

            for p in path_train:
                shutil.copy(p, os.path.join(path_out, 'TRAIN', lab))
                shutil.copy(p.split('.')[0] + extra_info_suf, os.path.join(path_out, 'TRAIN', lab))

            for p in path_val:
                shutil.copy(p, os.path.join(path_out, 'VAL', lab))
                shutil.copy(p.split('.')[0] + extra_info_suf, os.path.join(path_out, 'VAL', lab))


def create_dataset_per_labels_folder (images_folder_path, output_path, dataset_dict, labels, colorspace="RGB",
                           extra_info_suf=None, verbose=True):
    """
    This function gets the path of all images and create a dataset folder tree. It's better explain with an example.
    Suppose we have an folder with a bunch of images and a .csv file describing the labels for each of them. First,
    you can use the 'load_dataset_from_csv' to generate a dictionary of these images. Then, just call this function to
    create the folder organized.
    Before:
    root_folder:
        img1.png
        ...
        imgN.png
    After:
    root_folder:
        label1
            img1.png
            ...
            imgN1.png
        ...
        labelM
            img1.png
            ...
            imgNM.png

    If you have extra information, they also will be save along side the images.

    :param images_folder_path (string): the images' root folder
    :param output_path (string): the path where the dataset will take place
    :param dataset (dictionary): a dictionary in which a image path is a key and the label and extra information are
    the values. You can get this dict using the 'load_dataset_from_csv' function.
    :param labels (list): a list of string containing all dataset labels
    :param colorspace (string, optional): the following colorpace: ('RGB', 'HSV', 'Lab', 'XYZ', 'HSL' or 'YUV').
    Default is "RGB"
    :param extra_info_suf (string, optional): string containing the suffix if you have extra information. For example,
    "_info.txt". If None, there's no extra information. Default is None.
    :param verbose (bool, optional): if you wanna print information on the screen. Default is True

    """

    # First of all, let us create the folders
    create_folders(output_path, labels)
    dataset_size = len(dataset_dict)
    missing_imgs = list()




    for k, path in enumerate(dataset_dict):

        if (verbose):
            print ("Working on img {} of {}".format(k, dataset_size))

        try:

            # In this case we have extra information
            if (extra_info_suf is not None):
                label, extra_info = dataset_dict[path]
            else:
                label = dataset_dict[path]

            src_img_path = os.path.join(images_folder_path, path)
            dst_img_path = os.path.join(output_path, label, path)

            if (colorspace == "RGB"):
                shutil.copy(src_img_path, dst_img_path)
            else:
                img = convert_colorspace(src_img_path)
                cv2.imwrite(dst_img_path, img)

            if (extra_info_suf is not None):
                dst_extra_path = os.path.join(output_path, label, path.split(".")[0]) + extra_info_suf
                print (extra_info)
                np.savetxt(dst_extra_path, extra_info, fmt='%i', delimiter=',')


        except FileNotFoundError:
            if (verbose):
                print ('There is no image in the path {}. I am going forward and saving it in missing list'.format(path))
            missing_imgs.append(path)

        if (verbose):
            print("\n######## Summary ##########")
            print("Numeber of images of images: {}".format(dataset_size))
            print("Missed images: {}".format(len(missing_imgs)))
            print("\n###########################")



