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
from PIL import Image
from scipy.stats import friedmanchisquare, wilcoxon
from itertools import combinations
import random
import torch


def seed_everything(seed=8):
    """
    This function just seeds random intialization for everything you're going to use
    :param seed (int): a seed number
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def convert_colorspace(img_path, colorspace):
    """
    This function gets an RGB image path, load it and convert it to the desired colorspace

    :param img_path (string): the RGB image path
    :param colorspace (string): the following colorpace: ('HSV', 'Lab', 'XYZ', 'HSL' or 'YUV')
    :return (numpy.array): the converted image
    """

    img = cv2.imread(img_path)

    # In this case, the path is wrong
    if img is None:
        raise FileNotFoundError

    if colorspace == 'HSV':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif colorspace == 'Lab':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    elif colorspace == 'XYZ':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)
    elif colorspace == 'HLS':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    elif colorspace == 'YUV':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    else:
        raise Exception ("There is no conversion for {}".format(colorspace))

    return img


def plot_dataset_sample(dataset, n=6):
    """
    This function gets a pytorch dataset and sample from it to show n images.
    It's useful to check the augmentation in the images.
    :param dataset (torch.utils.dataset): a dataset containing images and labels only. Probably, the best way to load it
    is by using torchvision.datasets.ImageFolder()
    :param n (int): the number of times an image is sampled from the dataset
    """
    img = np.vstack((np.hstack(((dataset[i][0]).numpy().swapaxes(0, 2) for _ in range(n)))
                   for i in range(len(dataset))))
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def plot_img_data_loader (data_loader, n_img_row=4):
    """
    This function plots a set of images from a data loader
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

    :param img (np.array or PIL.Image.Image): the image or a grid of images (see plot_img_data_loader())
    :param grid (bool, optional): if you want to plot a grid, you can use torchvision.utils.make_grid or
     plot_img_data_loader() and set grid as True. Default is False
    :param title (string): the image plot title
    :param hit (bool, optional): if you set hit as True, the title will be plotted in green, False is red and None is
    Black. Default is None.
    :param save_path (string, optional): an string containing the full img path to save the plot in the disk. If None,
    the image is just plotted in the screen. Default is None.
    :param denorm (boolean): if you need to denorm the image set it as true
    """

    if denorm:
        img = denorm_img(img)

    if grid:
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    else:
        if type(img) == PIL.Image.Image:
            plt.imshow(np.array(img))
        else:
            npimg = img.numpy()
            if len(npimg.shape) == 4:
                npimg = npimg[0,:,:]

            img = np.moveaxis(npimg[:, :, :], 0, -1)
            plt.imshow(img)

    title_obj = plt.title(title, fontsize=8)
    if hit is None:
        plt.setp(title_obj, color='k')
    else:
        if hit:
            plt.setp(title_obj, color='g')
        else:
            plt.setp(title_obj, color='r')

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


def denorm_img (img, mean=(-0.485, -0.456, -0.406), std=(1/0.229, 1/0.224, 1/0.225)):
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


def shade_of_gray_cc(img, power=6, gamma=None):
    """
    This function applies the shadow of fray color constancy
    Original source: https://github.com/LincolnZjx/ISIC_2018_Classification/blob/master/previous_code/tf_version/color_constancy.py
    :param img (np.array): an image
    :param power (int): the degree of norm, 6 is used in reference paper
    :param gamma (float): the value of gamma correction, 2.2 is used in reference paper
    """

    img_dtype = img.dtype

    if gamma is not None:
        img = img.astype('uint8')
        look_up_table = np.ones((256,1), dtype='uint8') * 0
        for i in range(256):
            look_up_table[i][0] = 255 * pow(i/255, 1/gamma)
        img = cv2.LUT(img, look_up_table)

    img = img.astype('float32')
    img_power = np.power(img, power)
    rgb_vec = np.power(np.mean(img_power, (0,1)), 1/power)
    rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec/rgb_norm
    rgb_vec = 1/(rgb_vec*np.sqrt(3))
    img = np.multiply(img, rgb_vec)
	
    # This is to avoid strange colors on the images
    img = np.clip(img, a_min=0, a_max=255)

    return img.astype(img_dtype)


def apply_color_constancy_folder (input_folder_path, output_folder_path, img_exts=('jpg'), new_img_ext=None, resize=None):
    """
    This function applies the color constancy algorithm for a set of images in a folder
    :param input_folder_path (string): the path to the input folder
    :param output_folder_path (string): the path to the output folder
    :param img_exts (tuple): a list of image extensions to load
    :param new_img_ext (string): if you want a image extension different than the original one you just inform it here
    Ex: "png"
    :param resize (tuple): if you want to resize the original image, pass it here: Ex (224,224)
    """

    # Checking if the output_folder_path doesn't exist. If True, we must create it.
    if not os.path.isdir(output_folder_path):
        os.mkdir(output_folder_path)

    if new_img_ext is None:
        new_img_ext = img_exts[0]

    all_img_paths = list()
    for ext in img_exts:
        all_img_paths += (glob.glob(os.path.join(input_folder_path, '*.' + ext)))

    print("-" * 50)
    print("- Starting the color constancy process...")

    if len(all_img_paths) == 0:
        print("- There is no {} in {} folder".format(img_exts, input_folder_path))

    with tqdm(total=len(all_img_paths), ascii=True, ncols=100) as t:

        for img_path in all_img_paths:

            img_name = img_path.split('/')[-1]

            img_ = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if resize is not None:
                img_ = cv2.resize(img_, resize, cv2.INTER_AREA)

            np_img = shade_of_gray_cc (img_)
            cv2.imwrite(os.path.join(output_folder_path, img_name.split('.')[0] + '.' + new_img_ext), np_img)
            t.update()

    print("- All done!")
    print("-" * 50)


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


def insert_pred_col(data, labels_name, pred_pos=2, col_pred="PRED", output_path=None):
    """
    If the CSV/DF doesn't has a col_pred column, this function adds it based on the max prob. for the labels.
    :param data (string or pd.DataFrame): the predictions
    :param labels_name (list): the labels name
    :param pred_pos (int, optional): the position in which the col_pred will be added. Default is 2.
    :param col_pred (string, optional): the pred. name's column. Default is "PRED"
    :param output_path (string, optional): if you wanna save the dataframe on the disk, inform the path. Default is None
    :return: the dataframe with containing the PRED column
    """

    # If the data is a path, we load it.
    if isinstance(data, str):
        output_path = data
        data = pd.read_csv(data)

    # Checking if we need to include the prediction column or the DataFrame already has it.
    try:
        x = data[col_pred]
        return data
    except KeyError:
        print("- Inserting the pred column in DF...")
        new_cols = list(data.columns)
        new_cols.insert(pred_pos, col_pred)
        new_vals = list()
        for idx, row in data.iterrows():
            v = list(row.values)
            pred = labels_name[np.argmax(row[labels_name].values)]
            v.insert(pred_pos, pred)
            new_vals.append(v)

        data = pd.DataFrame(new_vals, columns=new_cols)
        if output_path is not None:
            data.to_csv(output_path, index=False)

        return data


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


def agg_models(ensemble, labels_name, image_name=None, agg_method="avg", output_path=None, ext_files="csv",
                 true_col="REAL", weigths=None):

    # Aggregation functions
    def avg_agg(df):
        return df.mean(axis=1)

    def max_agg(df):
        return df.max(axis=1)

    # If ensemble is a path, we need to load all files from a folder:
    all_data = list()
    if isinstance(ensemble, str):
        files = glob.glob(os.path.join(ensemble, "*." + ext_files))
        files.sort()
        for f in files:
            all_data.append(pd.read_csv(f))
    else:
        all_data = ensemble

    # Checking the weights if applicable
    if weigths is not None:
        if len(weigths) != len(all_data):
            raise ("You are using weights, so you must have one weight for each files in the folder")

        sum_w = sum(weigths)
        for idx in range(len(all_data)):
            all_data[idx][labels_name] = (all_data[idx][labels_name] * weigths[idx]) / sum_w


    # The list to store the values
    series_agg_list = list()
    labels_df = list()

    # Getting the ground true and images name (if applicable) and adding them to be included in the final dataframe

    try:
        if image_name is not None:
            s_img_name = all_data[0][image_name]
            series_agg_list.append(s_img_name)
            labels_df.append(image_name)
    except KeyError:
        print("Warning: There is no image_name! The code will run without it")

    try:
        if true_col is not None:
            s_true_labels = all_data[0][true_col]
            series_agg_list.append(s_true_labels)
            labels_df.append(true_col)
    except KeyError:
        print ("Warning: There is no true_col! The code will run without it")

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
        else:
            raise Exception("There is no {} aggregation method".format(agg_method))
        del series_label_list

    # Creating the dataframe and puting the labels name on it
    agg_df = pd.concat(series_agg_list, axis=1)
    agg_df.columns = labels_df
    if output_path is not None:
        agg_df.to_csv(output_path, index=False)

    return agg_df
