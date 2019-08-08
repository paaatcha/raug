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
from torchvision.utils import make_grid
import torchvision.transforms.functional as FTrans
import PIL
import pandas as pd
import glob
from tqdm import tqdm
from .color_constancy import shade_of_gray
from PIL import Image
from scipy.stats import friedmanchisquare, wilcoxon
from itertools import combinations
import shutil


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


def copy_imgs_from_list_to_folder (img_list, output_folder, base_path=None, ext=None):

    # Checking if the folder doesn't exist. If True, we must create it.
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    print ("-"*50)
    print ("- Starting to copy...")
    with tqdm(total=len(img_list), ascii=True, ncols=100) as t:
        for img in img_list:
            if base_path is not None:
                img = os.path.join(base_path, img)
            if ext is not None:
                img = img + "." + ext

            shutil.copy(img, output_folder)
            t.update()

    print("- All done!")
    print("-" * 50)


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
    """
    If you'd like to plot some images from a data loader, you should use this function
    :param data_loader (torchvision.utils.DataLoader): a data loader containing the dataset
    :param n_img_row (int, optional): the number of images to plot in the grid. Default is 4.
    """

    # Getting some samples to plot the images
    data_sample = iter(data_loader)
    data = data_sample.next()
    if (len(data) == 2):
        images, labels = data
    else:
        images, labels = data[0:2]

    # show images
    plot_img (make_grid(images, nrow=n_img_row, padding=10), True)


def plot_img (img, grid=False, title="Image test", hit=None, save_path=None, denorm=False):
    """
    This function plots one o more images on the screen.

    :param img (np.array or PIL.Image.Image): a image in a np.array or PIL.Image.Image.
    :param grid (bool, optional): if you'd like to post a grid, you can use torchvision.utils.make_grid and set grid as
    True. Default is False
    :param title (string): the image plot title
    :param hit (bool, optional): if you set hit as True, the title will be plotted in green, False is red and None is
    Black. Default is None.
    :param save_path (string, optional): an string containing the full img path to save the plot in the disk. If None,
    the image is just plotted in the screen. Default is None.
    """

    if denorm:
        img = denorm_img(img)

    if (grid):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    else:
        if (type(img) == PIL.Image.Image):
            plt.imshow(np.array(img))
        else:
            npimg = img.numpy()
            if (len(npimg.shape) == 4):
                npimg = npimg[0,:,:]

            img = np.moveaxis(npimg[:, :, :], 0, -1)
            plt.imshow(img)


    title_obj = plt.title(title, fontsize=8)
    if (hit is None):
        plt.setp(title_obj, color='k')
    else:
        if (hit):
            plt.setp(title_obj, color='g')
        else:
            plt.setp(title_obj, color='r')

    if (save_path is None):
        plt.show()
    else:
        plt.savefig(save_path)



def denorm_img (img, mean = (-0.485, -0.456, -0.406), std = (1/0.229, 1/0.224, 1/0.225)):
    """
    This function denormalize a given image
    :param img (torch.Tensor): the image to be denormalized
    :param mean (list or tuple): the mean to denormalize
    :param std (list or tuple): the std to denormalize
    :return: a tensor denormalized
    """

    img_inv = FTrans.normalize(img, [0.0,0.0,0.0], std)
    img_inv = FTrans.normalize(img_inv, mean, [1.0,1.0,1.0])

    return img_inv

def apply_color_constancy_folder (input_folder_path, output_folder_path, img_exts=['jpg']):

    # Checking if the output_folder_path doesn't exist. If True, we must create it.
    if (not os.path.isdir(output_folder_path)):
        os.mkdir(output_folder_path)

    all_img_paths = list()
    for ext in img_exts:
        all_img_paths += (glob.glob(os.path.join(input_folder_path, '*.' + ext)))

    print("-" * 50)
    print("- Starting the color constancy process...")

    if len(all_img_paths) == 0:
        print ("- There is no {} in {} folder".format(input_folder_path, img_exts))

    with tqdm(total=len(all_img_paths), ascii=True, ncols=100) as t:

        for img_path in all_img_paths:

            img_name = img_path.split('/')[-1]
            np_img = shade_of_gray (np.array(Image.open(img_path).convert("RGB")))
            img = Image.fromarray(np_img)
            img.save(os.path.join(output_folder_path, img_name))

            t.update()

    print("- All done!")
    print("-" * 50)


def merge_results_hierarchical(data, data_unk, output_path=None, cols=None, pred=False):
    if isinstance(data, str):
        data = pd.read_csv(data)

    if isinstance(data_unk, str):
        data_unk = pd.read_csv(data_unk)

    if cols is None:
        cols = ['image', 'MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']

    vals = list()
    for i_row_unk, i_row in zip(data_unk.iterrows(), data.iterrows()):
        new_row = list()
        idx_unk, row_unk = i_row_unk
        idx, row = i_row

        #        if row_unk['PRED'] == 'UNK':
        if row['image'] != row_unk['image']:
            raise ("The image name cannot be different!")
        #        print (row_unk['image'], idx_unk, row_unk['UNK'])
        #        print (row['image'], idx)
        #        print ("")
        for c in cols:
            if c == 'UNK':
                new_row.append(row_unk['UNK'])
            else:
                new_row.append(row[c])

        if pred:
            vp = cols[1:][np.argmax(new_row[1:])]
            new_row.insert(1, vp)

        vals.append(new_row)

    if pred:
        cols.insert(1, 'PRED')
    new_df = pd.DataFrame(vals, columns=cols)
    if output_path is None:
        new_df.to_csv("submission.csv", index=False)
    else:
        new_df.to_csv(output_path, index=False)

def get_all_prob_distributions (pred_csv_path, class_names, folder_path=None, plot=True):
    """
    This function gets a csv file containing the probabilities and ground truth for all samples and returns the probabi-
    lity distributions for each class.
    :param pred_csv_path (string): the full path to the csv file
    :param class_names (list): a list of string containing the class names
    :param save_path (string, optional): the folder you'd like to save the plots
    :return: it returns a tuple containing a list of the avg and std distribution for each class of the task
    """
    # Checking if the output_folder_path doesn't exist. If True, we must create it.
    if (not os.path.isdir(folder_path)):
        os.mkdir(folder_path)

    preds = pd.read_csv(pred_csv_path)

    # Getting all distributions
    distributions = list ()
    print ("Generating the distributions considering correct and incorrect classes:")
    for name in class_names:
        print ("For {}...".format(name))
        pred_label = preds[(preds['REAL'] == name)][class_names]
        full_path = os.path.join(folder_path, "{}_all_prob_dis.png".format(name))
        distributions.append(get_prob_distribution (pred_label, full_path, name, plot=plot))

    # Getting only the correct class
    correct_distributions = list()
    print("\nGenerating the distributions considering correct classes:")
    for name in class_names:
        print("For {}...".format(name))
        pred_label = preds[(preds['REAL'] == name)  & (preds['REAL'] == preds['PRED'])][class_names]
        full_path = os.path.join(folder_path, "{}_correct_prob_dis.png".format(name))
        correct_distributions.append(get_prob_distribution (pred_label, full_path, name, plot=plot))

    # Getting only the incorrect class
    correct_distributions = list()
    print("\nGenerating the distributions considering incorrect classes:")
    for name in class_names:
        print("For {}...".format(name))
        pred_label = preds[(preds['REAL'] == name) & (preds['REAL'] != preds['PRED'])][class_names]
        full_path = os.path.join(folder_path, "{}_incorrect_prob_dis.png".format(name))
        correct_distributions.append(get_prob_distribution(pred_label, full_path, name, plot=plot))

    return distributions, correct_distributions, correct_distributions



def get_prob_distribution (df_class, save_full_path=None, label_name=None, cols=None, plot=True):
    """
    This function generates and plot the probability distributions for a given dataframe containing the probabilities
    for each label in the classification problem.

    :param df_class (pd.dataframe or string): a pandas dataframe or a path to one, containing the probabilities and
    ground truth for each sample
    :param save_full_path (string, optional): the full path, including the image name, you'd like to plot the distribu-
    tion. If None, it'll be saved in the same folder you call the code. Default is None.
    :param label_name (string, optional): a string containing the name of the label you're generating the distributions.
    If None, it'll print Label in the plot. Default is None.
    :param cols (string, optional): if you're passing the path to a df_class you must say the columns you'd like
    to consider. If None, considers all columns

    :return: it returns a tuple containing the avg and std distribution.
    """

    # It's just to avoid the plt warning that we are plotting to many figures
    plt.rcParams.update({'figure.max_open_warning': 0})

    if isinstance (df_class, str):
        df_class = pd.read_csv(df_class)
        if cols is not None:
            df_class = df_class[cols]

    if save_full_path is None:
        save_full_path = "prob_dis.png"
    if label_name is None:
        label_name = "Label"

    avg = df_class.mean()
    std = df_class.std()

    if plot:
        ax = avg.plot(kind="bar", width=1.0, yerr=std)
        ax.grid(False)
        ax.set_title("{} prediction distribution".format(label_name))
        ax.set_xlabel("Labels")
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)

        plt.savefig(save_full_path, dpi=300)
        plt.figure()

    return avg, std


def agg_predictions(folder_path, labels_name, image_name=None, agg_method="avg", output_path=None,
                    ext_files="csv", true_col="REAL", weigths=None):
    """
    This function gets a folder path and aggregate all prediction inside this folder.
    :param folder_path:
    :param labels_name:
    :param agg_method:
    :param output_path:
    :param ext_files:
    :param true_col:
    :return:
    """
    # Aggregation functions
    def avg_agg(df):
        return df.mean(axis=1)

    def max_agg(df):
        return df.max(axis=1)

    # Getting all csv files in a folder
    files = glob.glob(os.path.join(folder_path, "*." + ext_files))
    files.sort()

    # Checking the weights if applicable
    if (weigths is not None) and (agg_method == "avg"):
        if len(weigths) != len(files):
            raise ("You are using weights, so you must have one weight for each files in the folder")
        sum_w = sum(weigths)

    # Loading the dataframes and applyting the weights if applicable
    all_data = list()
    for k, f in enumerate(files):
        df = pd.read_csv(f)
        if (weigths is not None) and (agg_method == "avg"):
            df[labels_name] = (df[labels_name] * weigths[k]) / sum_w
        all_data.append(df)


    # The list to store the values
    series_agg_list = list()
    labels_df = list()

    # Getting the ground true and images name (if applicable) and adding them to be included in the final dataframe
    if image_name is not None:
        s_img_name = all_data[0][image_name]
        series_agg_list.append(s_img_name)
        labels_df.append(image_name)

    if true_col is not None:
        s_true_labels = all_data[0][true_col]
        series_agg_list.append(s_true_labels)
        labels_df.append(true_col)

    for lab in labels_name:
        series_label_list = list()
        labels_df.append(lab)
        for data in all_data:
            series_label_list.append(data[lab])

        comb_df = pd.concat(series_label_list, axis=1)
        if agg_method == 'avg':
            series_agg_list.append(avg_agg(comb_df))
        elif agg_method == 'max':
            series_agg_list.append(max_agg(comb_df))
            pass
        elif agg_method == 'vote':
            # TODO: implement marjoritary vote agg
            pass
        else:
            raise ("There is no {} aggregation method".format(agg_method))
        del series_label_list

    # Creating the dataframe and puting the labels name on it
    agg_df = pd.concat(series_agg_list, axis=1)
    agg_df.columns = labels_df
    if output_path is not None:
        agg_df.to_csv(output_path, index=False)

    return agg_df


def statistical_test(data, names, pvRef, verbose=True):
    """
    This function performs the Friedman's test. If pv returned by the friedman test is lesser than the pvRef,
    the Wilcoxon test is also performed

    :param data: the data that the test will perform. Algorithms x Samples. for example a matrix 3 x 10 contains 3
    algorithms with 10 samples each
    :param names: a list with database names. Ex: n = ['X', 'Y', 'Z']
    :param pvRef: pvalue ref
    :param verbose: set true to print the resut on the screen
    :return:
    """
    data = np.asarray(data)
    if data.shape[0] != len(names):
        raise ('The size of the data row must be the same of the names')

    out = 'Performing the Friedman\'s test...'
    sFri, pvFri = friedmanchisquare(*[data[i, :] for i in range(data.shape[0])])
    out += 'Pvalue = ' + str(pvFri) + '\n'

    if pvFri > pvRef:
        out += 'There is no need to pairwise comparison because pv > pvRef'
    else:
        out += '\nPerforming the Wilcoxon\'s test...\n'
        combs = list(combinations(range(data.shape[0]), 2))
        for c in combs:
            sWil, pvWill = wilcoxon(data[c[0], :], data[c[1], :])
            out += 'Comparing ' + names[c[0]] + ' - ' + names[c[1]] + ': pValue = ' + str(pvWill) + '\n'

    if verbose:
        print (out)

    return out