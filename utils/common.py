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

    print ("Starting the color constancy process...")
    with tqdm(total=len(all_img_paths), ascii=True, ncols=100) as t:

        for img_path in all_img_paths:

            img_name = img_path.split('/')[-1]
            np_img = shade_of_gray (np.array(Image.open(img_path).convert("RGB")))
            img = Image.fromarray(np_img)
            img.save(os.path.join(output_folder_path, img_name))

            t.update()


    print (len(all_img_paths))


def get_all_prob_distributions (pred_csv_path, class_names, folder_path=None):
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
        distributions.append(get_prob_distribution (pred_label, full_path, name))

    # Getting only the correct class
    correct_distributions = list()
    print("\nGenerating the distributions considering correct classes:")
    for name in class_names:
        print("For {}...".format(name))
        pred_label = preds[(preds['REAL'] == name)  & (preds['REAL'] == preds['PRED'])][class_names]
        full_path = os.path.join(folder_path, "{}_correct_prob_dis.png".format(name))
        correct_distributions.append(get_prob_distribution (pred_label, full_path, name))

    # Getting only the incorrect class
    correct_distributions = list()
    print("\nGenerating the distributions considering incorrect classes:")
    for name in class_names:
        print("For {}...".format(name))
        pred_label = preds[(preds['REAL'] == name) & (preds['REAL'] != preds['PRED'])][class_names]
        full_path = os.path.join(folder_path, "{}_incorrect_prob_dis.png".format(name))
        correct_distributions.append(get_prob_distribution(pred_label, full_path, name))

    return distributions, correct_distributions, correct_distributions



def get_prob_distribution (df_class, save_full_path=None, label_name=None, cols=None):
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

    # Working on AK
    avg = df_class.mean()
    std = df_class.std()

    ax = avg.plot(kind="bar", width=1.0, yerr=std)
    ax.grid(False)
    ax.set_title("{} prediction distribution".format(label_name))
    ax.set_xlabel("Labels")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)

    plt.savefig(save_full_path, dpi=300)
    plt.figure()

    return avg, std