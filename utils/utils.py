#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: André Pacheco
E-mail: pacheco.comp@gmail.com

This file contains auxiliary functions to handle to data in jedy library

If you find any bug or have some suggestion, please, email me.
"""

import os
import glob
from random import shuffle, seed


def create_folders(path, folders=['A', 'B'], train_test_val=False):
    """
    This function creates a folder tree inside a root folder's path informed as parameter.

    Parameters:
    path (string): the root folder's path
    folders (list): a list of strings representing the name of the folders will be created inside the root.
    Default is ['A', 'B']
    train_test_val (bool): if you wanns create TRAIN, TEST and VAL partition for each folder. Default is false.
    """

    # Checking if the folder doesn't exist. If True, we must create it.
    if (not os.path.isdir(path)):
        os.mkdir(path)

    if (train_test_val):
        if (not os.path.isdir(path + '/' + 'TEST')):
            os.mkdir(path + '/' + 'TEST')
        if (not os.path.isdir(path + '/' + 'TRAIN')):
            os.mkdir(path + '/' + 'TRAIN')
        if (not os.path.isdir(path + '/' + 'VAL')):
            os.mkdir(path + '/' + 'VAL')

    for folder in folders:
        if (train_test_val):
            if (not os.path.isdir(path + '/TRAIN/' + folder)):
                os.mkdir(path + '/TRAIN/' + folder)
            if (not os.path.isdir(path + '/TEST/' + folder)):
                os.mkdir(path + '/TEST/' + folder)
            if (not os.path.isdir(path + '/VAL/' + folder)):
                os.mkdir(path + '/VAL/' + folder)
        else:
            if (not os.path.isdir(path + '/' + folder)):
                os.mkdir(path + '/' + folder)


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

    Parameters:
    path_in (string): the root folder path that you wanna split in the train, test and val partitions
    path_out (string): the root folder path that will receive the new folder structure
    extra_info_suf (string): if the images have an extra information file you must inform the suffix
    (see get_path_from_folders in loader.py for more information). Default is None.
    sets_perc (list): the set percentage of data for [train, val, test]. It must sum up 1.0. Default is [0.8, 0.1, 0.1]
    shuf (bool): set it as True if you wanna shuffle the images. Default is True.
    seed_number (int): the seed number to keep the shuffle for multiples executions. Default is None.
    verbose (bool): set it as True to print information on the screen. Default is True.

    """

    # Checking the % for the partitions
    if (abs(sum(sets_perc)) >= 0.01):
        print('The % in sets_perc must sum up 1.0')
        raise ValueError

    # Getting all folders in the root folder path
    folders = [nf for nf in os.listdir(path_in) if os.path.isdir(os.path.join(path_in, nf))]

    # Calling create folders to create the new folder structure considering Train, Val and Test partitions
    create_folders(path_out, folders, True)

    for lab in folders:
        path_imgs = glob.glob(path_in + '/' + lab + '/*.' + img_ext)

        if shuf:
            # This is used to keep the same partitions for each train, val and test sets
            if (seed_number is not None):
                seed(seed_number)
            shuffle(path_imgs)

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
                os.system('cp ' + p + ' ' + path_out + '/TEST/' + lab)

            for p in path_train:
                os.system('cp ' + p + ' ' + path_out + '/TRAIN/' + lab)

            for p in path_val:
                os.system('cp ' + p + ' ' + path_out + '/VAL/' + lab)
        else:
            for p in path_test:
                os.system('cp ' + p + ' ' + path_out + '/TEST/' + lab)
                os.system('cp ' + p.split('.')[0] + extra_info_suf + ' ' + path_out + '/TEST/' + lab)

            for p in path_train:
                os.system('cp ' + p + ' ' + path_out + '/TRAIN/' + lab)
                os.system('cp ' + p.split('.')[0] + extra_info_suf + ' ' + path_out + '/TRAIN/' + lab)

            for p in path_val:
                os.system('cp ' + p + ' ' + path_out + '/VAL/' + lab)
                os.system('cp ' + p.split('.')[0] + extra_info_suf + ' ' + path_out + '/VAL/' + lab)
