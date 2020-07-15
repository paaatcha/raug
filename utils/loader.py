#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: André Pacheco
E-mail: pacheco.comp@gmail.com

This file contains functions and auxiliary functions to load and handle a dataset to train using raug

If you find any bug or have some suggestion, please, email me.
"""

from random import seed
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from glob import glob
import os

def parse_metadata(data_csv, replace_nan=None, cols_to_parse=None, replace_rules=None, save_path=None):
    """
    This function parses the metadata within a csv/dataframe in order to transform string data in one_hot_encode
    information. For example, if you have a column Gender that assume values as 'Female' or 'Male', this code will
    create a new dataframe that replaces the column Gender to two columns: 'Female' and 'Male', in which will assume 1
    or 0, depending on the gender.

    :param data_csv (string or pandas.dataframe): the path for a csv or a dataframe already loaded
    :param replace_nan (string or int or float or boolean or None): if you have NaN or missing data in your dataset you
    can use this variable to replace them to a value. However, if you set it as None, the missing/NaN data will be
    removed from the dataset. Default is None.
    :param cols_to_parse (list, optional): a list of strings containing the column names to be parsed. If it's None,
    none column will be parsed to hot encode. Default is None.
    :param replace_rules (dict, optional): If you'd like to replace data in any column of your dataset, you can define
    this rules here. For example, supose I have a column call 'A' that assume values as 1 or 'b'. If you'd like to
    replace every incidence of 'b' to 2, you must do: replace_rules = {'A': {'b':2}}. Obviously, if you wanna insert
    more rules for the same column or for a different one, you just need to follow the pattern. If None, none rule will
    be carried out. Default is None.
    :param save_path (string, optional): if you want to save the final dataframe, just set the path. If None, the
    dataframe won't be save. Default is None.
    :return: a parsed pandas.dataframe
    """

    print("-" * 50)
    print("- Parsing the input csv...")

    # Reading the csv
    if isinstance(data_csv, str):
        data = pd.read_csv(data_csv)
    else:
        data = data_csv

    # Checking if we need to remove any possible NaN
    if replace_nan is None:
        data = data.dropna()
    else:
        data = data.fillna(replace_nan)


    # If replace rules is true, we'll just replace the values by the given
    # rules dictionary
    if replace_rules is not None:
        cols_rules_keys = list(replace_rules.keys())
        for col in cols_rules_keys:
            col_rules = list(replace_rules[col].keys())
            col_values = list(replace_rules[col].values())
            try:
                data[col] = data[col].replace(col_rules, col_values)
            except TypeError:
                pass

    # If cols_to_handle is None the function will return the data without do anything
    new_col_names = list()
    if cols_to_parse is not None:

        for col in cols_to_parse:
            # Getting the unique values
            col_values = list(data[col].unique())

            # Removing all elements containing the replace_nan
            if replace_nan is not None:
                try:
                    col_values.remove(replace_nan)
                except ValueError:
                    pass

            # Sorting the values and saving them in new_col_names
            col_values.sort()
            new_col_names += col_values

        # Now, let's train to compose the new dataframe:
        # Getting the original col values and removing the handle ones
        original_col_names = list(data.columns.values)
        for c in cols_to_parse:
            original_col_names.remove(c)
            # Putting together the original and new columns. Now we have our final dataframe names
        data_col_names = original_col_names + new_col_names
        original_col_names = list(data.columns.values)

        # Now, let's iterate through the old data and get all values for each sample, replacing
        # for the one_hot encode if applicable
        values = list()
        for idx, row in data.iterrows():

            row_dict = {c: 0 for c in data_col_names}
            for col in original_col_names:
                if col in data_col_names:
                    row_dict[col] = row[col]
                else:
                    if replace_nan is not None:
                        if row[col] == replace_nan:
                            row_dict[row[col]] = 0
                        else:
                            row_dict[row[col]] = 1
                    else:
                        row_dict[row[col]] = 1

            values.append(row_dict)

        data = pd.DataFrame(values, columns=data_col_names)

    if save_path is not None:
        data.to_csv(save_path, columns=data_col_names, index=False)

    print("- csv parsed!")
    print("-" * 50)
    return data


def split_train_val_test_csv (data_csv, save_path=None, tr=0.80, tv=0.10, te=0.10, seed_number=None):
    """
    This function gets a csv/dataframe and returns it with a new column called 'partition' with the train, val, and test
    partitions to train a classifier
    :param data_csv (string or pd.DataFrame): the path for a csv or a dataframe already loaded
    :param save_path (string): the path to save the result of this function. Default is None
    :param tr (number, optional): the % of data to share between train and val partitions. Default is 0.8
    :param te (number, optional): the % of data to use in the test partition. Default is 0.10
    :param tv (number, optional): the % of data to use in the validation partition. Default is 0.10
    :param seed_number (number, optional): a seed number to guarantee reproducibility
    return (pd.DataFrame): the dataframe with the new column
    """
    # Loading the data_csv
    if isinstance(data_csv, str):
        data_csv = pd.read_csv(data_csv)

    # Checking the % for the partitions
    if abs(1.0 - tr - te - tv) >= 0.01:
        raise Exception('The values of tr and te must sum up 1.0')

    # Setting the seed to reproduce the results later
    if seed_number is not None:
        seed(seed_number)
        np.random.seed(seed_number)

    # shuffling data
    data_csv = data_csv.sample(frac=1, random_state=seed_number).reset_index(drop=True)
    data_csv['partition'] = None

    # Getting the partitions numbers
    N = len(data_csv)
    n_test = int(round(te * N))
    n_val = int(round(tv * N))
    n_train = N - n_test - n_val

    data_csv.loc[0:n_test, 'partition'] = 'test'
    data_csv.loc[n_test:(n_test + n_val), 'partition'] = 'val'
    data_csv.loc[(n_test + n_val):(n_test + n_val + n_train), 'partition'] = 'train'

    if save_path is not None:
        data_csv.to_csv(save_path, index=False)

    return data_csv


def split_k_folder_csv (data_csv, col_target, save_path=None, k_folder=5, seed_number=None):
    """
    This function gets a csv/dataframe and creates a new column called 'folder' that represents the k-folder cross
    validation
    :param data_csv (string or pd.DataFrame): the path for a csv or a dataframe already loaded
    :param col_target (string): the name of the target/label column
    :param k_folder(int): the number of folders for the cross validation
    :param save_path (string): the path to save the result of this function. Default is None
    :param seed_number (number, optional): a seed number to guarantee reproducibility
    return (pd.DataFrame): the dataframe with the new column
    """

    print("-" * 50)
    print("- Generating the {}-foders...".format(k_folder))

    # Loading the data_csv
    if isinstance(data_csv, str):
        data_csv = pd.read_csv(data_csv)

    skf = StratifiedKFold(k_folder, True, seed_number)
    target = data_csv[col_target]
    data_csv['folder'] = None
    for folder_number, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(target)), target)):
        data_csv.loc[val_idx, 'folder'] = folder_number + 1

    if save_path is not None:
        data_csv.to_csv(save_path, index=False)

    print("- Done!")
    print("-" * 50)

    return data_csv


def create_csv_from_folders (base_path, img_exts=('png'), save_path=None, img_id="Image_id", target="target"):
    """
    This function creates a csv file from a dataset structered in a folder tree format
    :param base_path (string): the path to the dataset
    :param img_exts (list, tuple): a list with image extensions to load from the folders
    :param save_path (string, optional): the path to save the csv file. Default is None
    return: a dataframe with the image id and label
    """

    folder_names = [nf for nf in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, nf))]

    if (len(folder_names) == 0):
        folders = glob(base_path)
    else:
        folders = glob(os.path.join(base_path, '*'))

    paths, labels = list(), list()
    for folder in folders:
        lab = folder.split('/')[-1]
        for ext in img_exts:
            _paths = (glob(os.path.join(folder, '*.' + ext)))
            paths += [p.split('/')[-1] for p in _paths]
            labels += [lab] * len(_paths)

    if (len(paths) == 0):
        raise Exception("There is no image with the extensions {} in the given path".format(img_exts))

    data_csv = pd.DataFrame(list(zip(paths, labels)), columns=[img_id, target])

    if save_path is not None:
        data_csv.to_csv(save_path, index=False)

    return data_csv


def label_categorical_to_number (data_csv, col_target, col_target_number=None, save_path=None):
    """
    This function converts a label from categorical to number. The values are converted to code in alphabetic ordem.
    Example: for a set of labels ['A', 'B', 'C'] it converts to [0, 1, 2].
    :param data_csv (string or pd.DataFrame): the path for a csv or a dataframe already loaded
    :param col_target (string): the name of the target/label column
    :param col_target_number (string): if you want to control the name of the column with the convert number, just set
    the name here, otherwise it will set <col_target>_number
    :param save_path (string): the path to save the result of this function. Default is None
    return: it returns the same dataframe with an additional column called <col_target>_number or col_target_number
    """

    # Loading the data_csv
    if isinstance(data_csv, str):
        data_csv = pd.read_csv(data_csv)

    data_csv[col_target] = data_csv[col_target].astype('category')
    if col_target_number is None:
        data_csv[col_target + '_number'] = data_csv[col_target].cat.codes
    else:
        data_csv[col_target_number] = data_csv[col_target].cat.codes

    if save_path is not None:
        data_csv.to_csv(save_path, index=False)

    return data_csv


def get_labels_frequency (data_csv, col_target, col_single_id, verbose=False):
    """
    This function returns the frequency of each label in the dataset
    :param data_csv (string or pd.DataFrame): the path for a csv or a dataframe already loaded
    :param col_target (string): the name of the target/label column
    :param col_single_id (string): the name any column that is present for all rows in the dataframe
    :param verbose (boolean): a boolean to print or not the frequencies
    return (pd.DataFrame): a dataframe containing the frequency of each label
    """

    # Loading the data_csv
    if isinstance(data_csv, str):
        data_csv = pd.read_csv(data_csv)

    data_ = data_csv.groupby([col_target])[col_single_id].count()
    if (verbose):
        print('### Data summary: ###')
        print(data_)
        print(">> Total samples: {} <<".format(data_.sum()))

    return data_
