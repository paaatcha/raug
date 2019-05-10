#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: André Pacheco
E-mail: pacheco.comp@gmail.com

This file function to evaluate a model

If you find any bug or have some suggestion, please, email me.
"""

import os
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from ..model.metrics import Metrics, AVGMetrics, accuracy
from ..model.checkpoints import load_model
from ...utils import common
from tqdm import tqdm



def metrics_for_eval (model, data_loader, device, loss_fn, topk=2):
    """
        This function returns accuracy, topk accuracy and loss for the evaluation partition

        :param model (nn.Model): the model you'd like to evaluate
        :param data_loader (DataLoader): the DataLoader containing the data partition
        :param checkpoint_path(string, optional): string with a checkpoint to load the model. If None, none checkpoint is
        loaded. Default is None.
        :param loss_fn (nn.Loss): the loss function used in the training

        :return: a instance of the classe metrics
    """

    # setting the model to evaluation mode
    model.eval()

    print ("\nEvaluating...")
    # Setting tqdm to show some information on the screen
    with tqdm(total=len(data_loader), ascii=True, ncols=100) as t:

        # Setting require_grad=False in order to dimiss the gradient computation in the graph
        with torch.no_grad():

            # batch_time = AverageMeter()
            # data_time = AverageMeter()
            loss_avg = AVGMetrics()
            acc_avg = AVGMetrics()
            topk_avg = AVGMetrics()


            for data in data_loader:

                # In data we may have imgs, labels and extra info. If extra info is [], it means we don't have it
                # for the this training case. Imgs came in data[0], labels in data[1] and extra info in data[2]
                images_batch, labels_batch, extra_info_batch = data
                if (len(extra_info_batch)):
                    # Moving the data to the deviced that we set above
                    images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)
                    extra_info_batch = extra_info_batch.to(device)

                    # Doing the forward pass using the extra info
                    pred_batch = model(images_batch, extra_info_batch)
                else:
                    # Moving the data to the deviced that we set above
                    images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)

                    # Doing the forward pass without the extra info
                    pred_batch = model(images_batch)

                # Computing the loss
                L = loss_fn(pred_batch, labels_batch)
                # Computing the accuracy
                acc, topk_acc = accuracy(pred_batch, labels_batch, topk=(1, topk))

                loss_avg.update(L.item())
                acc_avg.update(acc.item())
                topk_avg.update(topk_acc.item())

                # Updating tqdm
                t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
                t.update()


    return {"loss": loss_avg(), "accuracy": acc_avg(), "topk_acc": topk_avg() }


# Testing the model
def test_model (model, data_loader, checkpoint_path= None, loss_fn=None, device=None,
                    partition_name='Test', metrics=['accuracy'], class_names=None, metrics_options=None,
                    verbose=True):
    """
    This function evaluates a given model for a given data_loader

    :param model (nn.Model): the model you'd like to evaluate
    :param data_loader (DataLoader): the DataLoader containing the data partition
    :param checkpoint_path(string, optional): string with a checkpoint to load the model. If None, none checkpoint is
    loaded. Default is None.
    :param loss_fn (nn.Loss): the loss function used in the training
    :param partition_name (string): the partition name
    :param metrics (list, tuple or string): it's the variable that receives the metrics you'd like to compute. Default
        is only the accuracy.
    :param class_names (list, tuple): a list or tuple containing the classes names in the same order you use in the
        label. For ex: ['C1', 'C2']. For more information about the options, please, refers to
        jedy.pytorch.model.metrics.py
    :param metrics_ options (dict): this is a dict containing some options to compute the metrics. Default is None.
    For more information about the options, please, refers to jedy.pytorch.model.metrics.py
    :param device (torch.device, optional): the device to use. If None, the code will look for a device. Default is
    None. For more information about the options, please, refers to jedy.pytorch.model.metrics.py
    :param verbose (bool, optional): if you'd like to print information o the screen. Default is True

    :return: a instance of the classe metrics
    """

    if (checkpoint_path is not None):
        model = load_model(checkpoint_path, model)

    # setting the model to evaluation mode
    model.eval()

    if (device is None):
        # Setting the device
        if torch.cuda.is_available():
            device = torch.device("cuda:" + str(torch.cuda.current_device()))
        else:
            device = torch.device("cpu")

    # Moving the model to the given device
    model.to(device)

    if (loss_fn is None):
        loss_fn = nn.CrossEntropyLoss()

    # Setting require_grad=False in order to dimiss the gradient computation in the graph
    with torch.no_grad():

        n_samples = len(data_loader)
        loss_avg = 0

        # Setting the metrics object
        metrics = Metrics (metrics, class_names, metrics_options)

        print("Testing...")
        # Setting tqdm to show some information on the screen
        with tqdm(total=len(data_loader), ascii=True, ncols=100) as t:

            for data in data_loader:

                # In data we may have imgs, labels and extra info. If extra info is [], it means we don't have it
                # for the this training case. Imgs came in data[0], labels in data[1] and extra info in data[2]
                images_batch, labels_batch, extra_info_batch = data
                if (len(extra_info_batch)):
                    # Moving the data to the deviced that we set above
                    images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)
                    extra_info_batch = extra_info_batch.to(device)

                    # Doing the forward pass using the extra info
                    pred_batch = model(images_batch, extra_info_batch)
                else:
                    # Moving the data to the deviced that we set above
                    images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)

                    # Doing the forward pass without the extra info
                    pred_batch = model(images_batch)

                # Computing the loss
                L = loss_fn(pred_batch, labels_batch)
                loss_avg += L

                # Moving the data to CPU and converting it to numpy in order to compute the metrics
                pred_batch_np = pred_batch.cpu().numpy()
                labels_batch_np = labels_batch.cpu().numpy()

                # updating the scores
                metrics.update_scores(labels_batch_np, pred_batch_np)

                # Updating tqdm
                t.set_postfix(loss='{:05.3f}'.format(loss_avg))
                t.update()

        # Getting the loss average
        loss_avg = loss_avg / n_samples

        # Getting the metrics
        metrics.compute_metrics()

        # Adding loss into the metric values
        metrics.add_metric_value("loss", loss_avg)

    if (verbose):
        print('- {} metrics:'.format(partition_name))
        metrics.print()


    return metrics.metrics_values


def visualize_model (model, data_loader, class_names, n_imgs=8, checkpoint_path=None, device_name="cpu",
                     save_path=None, topk=None):
    """
    This functino gets a dataset, provide the predctions on it and plot/save the images showing their labels and
    probabilities

    :param model (torch.nn.Module): a model to be used as predictor
    :param data_loader (torch.utils.DataLoader): a dataloader containing the dataset to be evaluate
    :param class_names (list): a list of strings containing the class names
    :param n_imgs (int or string, optional): the number of images to be plotted or saved. If you pass 'all', it will
    evaluate all images in the dataloader, no matter how many it is. Default is 8.
    :param checkpoint_path (string, optional): if you'd lije to load the model from a saved checkpoint, you need to pass
    the full path of this file. If None, the current model is used. Default is None.
    :param device_name (string, optional): the device you'd like to use to execute the evaluation. Default is "cpu"
    :param save_path (string, optional): if you'd like to save the images instead plot in the screen, just passa the
    folder path in which the images should be saved
    :param topk (int, optional): if you'd like to plot the topk predictions in the image's title, se the topk here. If
    None, the title will be only the pred and true class. Default is None.
    """

    # Loading the model
    if (checkpoint_path is not None):
        model = load_model(checkpoint_path, model)

    # setting the model to evaluation mode
    model.eval()

    # setting the device
    device = torch.device(device_name)

    # Moving the model to the given device
    model.to(device)

    # If it's None, we're going to run for the whole data_loader
    if (n_imgs == 'all'):
        n_imgs = len(data_loader) * data_loader.batch_size

    plotted_imgs = 0

    # Setting require_grad=False in order to dimiss the gradient computation in the graph
    with torch.no_grad():

        for data in data_loader:

            # In data we may have imgs, labels and extra info. If extra info is [], it means we don't have it
            # for the this training case. Imgs came in data[0], labels in data[1] and extra info in data[2]
            images_batch, labels_batch, extra_info_batch = data
            if (len(extra_info_batch)):
                # Moving the data to the deviced that we set above
                images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)
                extra_info_batch = extra_info_batch.to(device)

                # Doing the forward pass using the extra info
                pred_batch = model(images_batch, extra_info_batch)
            else:
                # Moving the data to the deviced that we set above
                images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)

                # Doing the forward pass without the extra info
                pred_batch = model(images_batch)

            # Getting the label
            _, pred = torch.max(pred_batch.data, 1)

            if (topk is not None):
                # As we are using the LogSoftmax, if we apply the exp we get the probs
                pred_prob = torch.exp(pred_batch)

                # Finding the topk predictions
                topk_pred_prob, topk_pred_label = pred_prob.topk(topk, dim=1)

            if (device_name is not "cpu"):
                # Moving the data to CPU and converting it to numpy in order to compute the metrics
                pred_batch_np = pred.cpu().numpy()
                labels_batch_np = labels_batch.cpu().numpy()
                images_batch_np = images_batch.cpu().numpy()

                if (topk is not None):
                    topk_pred_prob_np = topk_pred_prob.cpu().numpy()
                    topk_pred_label_np = topk_pred_label.cpu().numpy()
            else:
                pred_batch_np = pred.numpy()
                labels_batch_np = labels_batch.numpy()
                images_batch_np = images_batch.numpy()

                if (topk is not None):
                    topk_pred_prob_np = topk_pred_prob.numpy()
                    topk_pred_label_np = topk_pred_label.numpy()

            # Getting only the number of images
            if (n_imgs is not None):
                pred_batch_np = pred_batch_np[0:n_imgs]
                labels_batch_np = labels_batch_np[0:n_imgs]
                images_batch_np = images_batch[0:n_imgs, :, :, :]

                if (topk is not None):
                    topk_pred_label_np = topk_pred_label_np[0:n_imgs, :]
                    topk_pred_prob_np = topk_pred_prob_np[0:n_imgs, :]

            for k in range(pred_batch_np.shape[0]):

                title = "true: {} - pred: {}".format(class_names[labels_batch_np[k]], class_names[pred_batch_np[k]])
                hit = pred_batch_np[k] == labels_batch_np[k]

                if (topk is not None):
                    title+= "\n| "
                    for n in range(topk):
                        title+= "{}: {:.2f}% | ".format(class_names[topk_pred_label_np[k,n]], topk_pred_prob_np[k, n] * 100)


                if (save_path is not None):
                    img_path_name = os.path.join(save_path, "img_{}.png".format(plotted_imgs))
                    common.plot_img(images_batch_np[k], title=title, hit=hit, save_path=img_path_name)
                else:
                    common.plot_img(images_batch_np[k], title=title, hit=hit)

                plotted_imgs +=1
                print ("Plotting image {} ...".format(plotted_imgs))

            if (plotted_imgs >= n_imgs):
                break


def predict (img_path, model, class_names, extra_info=None, size=None, checkpoint_path=None, device_name="cpu",
             topk=None, normalize=None):
    """
    This function gets an image path, and provide its prediction, plot and its label probability.

    :param img_path (string): the image full path
    :param model (torch.nn.Module): a model to be used as predictor
    :param class_names (list): a list of strings containing the class names
    :param n_imgs (int or string, optional): the number of images to be plotted or saved. If you pass 'all', it will
    evaluate all images in the dataloader, no matter how many it is. Default is 8.
    :param extra_info (np.array, optional): if you have extra information to be loaded along with the image, you need
    to pass it. Default is None.
    :param size: (list or tuple, optional): if you need to resize the image to adequate the model input size, pass
    the size like (w, h). If None, the image won't be resized. Defaut is None.
    :param checkpoint_path (string, optional): if you'd lije to load the model from a saved checkpoint, you need to pass
    the full path of this file. If None, the current model is used. Default is None.
    :param device_name (string, optional): the device you'd like to use to execute the evaluation. Default is "cpu"
    :param save_path (string, optional): if you'd like to save the images instead plot in the screen, just passa the
    folder path in which the images should be saved
    :param topk (int, optional): if you'd like to plot the topk predictions in the image's title, se the topk here. If
    None, the title will be only the pred and true class. Default is None.
    :param normalize (2d list, optional): If you need to normaize your image you must pass a a 2d list containing the
    mean and std to normalize a tensor image. Ex: supposing we have n is the number of channels,
    normalize = [[m1, m2, ..., mn], [s1, s2, ..., sn]]. The operation the function will carry out is:
    input[channel] = (input[channel] - mean[channel]) / std[channel]. If None, the image will nome be normalized.
    Default is None.
    """

    img = Image.open(img_path)

    # Resizing if needed
    if (size is not None):
        img = img.resize(size)

    # Convert to numpy, transpose color dimension and normalize
    img = np.array(img).transpose((2, 0, 1)) / 256

    # Normalizing, if needed
    if (normalize is not None):
        means = np.array(normalize[0]).reshape((3, 1, 1))
        stds = np.array(normalize[1]).reshape((3, 1, 1))
        img = img - means
        img = img / stds

    # Converting to tensor
    img_tensor = torch.Tensor(img).view(-1, 3, size[0], size[0])
    # img_tensor = torch.Tensor(img)

    # print (img_tensor.shape)

    # Loading the model
    if (checkpoint_path is not None):
        model = load_model(checkpoint_path, model)

    # setting the model to evaluation mode
    model.eval()

    # setting the device
    device = torch.device(device_name)

    # Moving the model to the given device
    model.to(device)

    with torch.no_grad():

        # Getting one batch considering if we have the extra information
        if (extra_info is None):
            pred_scores = model(img_tensor)
        else:
            pred_scores = model(img_tensor, extra_info)

        # Getting the label
        _, pred_label = torch.max(pred_scores.data, 1)

        # As we are using the LogSoftmax, if we apply the exp we get the probs
        pred_probs = torch.exp(pred_scores)

        if (topk is not None):
            # Finding the topk predictions
            topk_pred_probs, topk_pred_label = pred_probs.topk(topk, dim=1)

        if (device_name is not "cpu"):
            # Moving the data to CPU and converting it to numpy in order to compute the metrics
            pred_label_np = pred_label.cpu().numpy()
            pred_probs_np = pred_probs.cpu().numpy()

            if (topk is not None):
                topk_pred_probs_np = topk_pred_probs.cpu().numpy()
                topk_pred_label_np = topk_pred_label.cpu().numpy()
        else:
            pred_label_np = pred_label.numpy()
            pred_probs_np = pred_probs.numpy()

            if (topk is not None):
                topk_pred_probs_np = topk_pred_probs.numpy()
                topk_pred_label_np = topk_pred_label.numpy()


        title = "Pred: {}".format(class_names[pred_label_np[0]])

        if (topk is not None):
            title += "\n| "
            for n in range(topk):
                title += "{}: {:.2f}% | ".format(class_names[topk_pred_label_np[0,n]], topk_pred_probs_np[0,n] * 100)

        common.plot_img(img_tensor, title=title, hit=None)


    return pred_label_np, pred_probs_np

        
        



