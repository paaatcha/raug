#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: André Pacheco
E-mail: pacheco.comp@gmail.com

This file implements some functions to perform classification metrics

If you find any bug or have some suggestion, please, email me.
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sklearn.metrics as skmet
import matplotlib.pyplot as plt
import numpy as np
from .common import one_hot_encoding
import itertools


def _check_dim (lab_real, lab_pred, mode='labels'):
    """
    This  function checks if y_real and y_pred are 1-d or 2-d. If mode is 'labels', this function returns an 1-d array
    ints, for ex: [0, 0, 1, 1, 2], in which each number is a label. If mode is 'scores', this function returns an 2-d
    array following the one hot encoding methos. For ex: [[0,0,0], [0,0,0], [0,1,0], [0,1,0], [0,0,1]]

    :param lab_real(1d or 2d n.array): the data real labels
    :param lab_pred(1d or 2d n.array): the predictions returned by the model
    :param mode (string, optional): the operation mode described above. Default is 'labels'
    :return (np.array, np.array): returns the lab_real and lab_pred transformed
    """
    if (mode == 'labels'):
        if (lab_real.ndim == 2):
            lab_real = lab_real.argmax(axis=1)
        if (lab_pred.ndim == 2):
            lab_pred = lab_pred.argmax(axis=1)

    elif (mode == 'scores'):
        if (lab_real.ndim == 1):
            lab_real = one_hot_encoding(lab_real)
        if (lab_pred.ndim == 1):
            lab_pred = one_hot_encoding(lab_pred)

    else:
        raise Exception ('There is no mode called {}. Please, choose between score or labels'.format(mode))

    return lab_real, lab_pred


def accuracy (lab_real, lab_pred, verbose=False):
    """
    Computess the accuracy. Both lab_real and lab_pred can be a labels array or and a array of
    scores (one hot encoding) for each class.

    :param lab_real(np.array): the data real labels
    :param lab_pred(np.array): the predictions returned by the model
    :param verbose(bool, optional): if you'd like to print the accuracy. Dafault is False.
    :return (float): the accuracy
    """

    # Checkin the array dimension
    lab_real, lab_pred = _check_dim (lab_real, lab_pred, mode='labels')

    acc = skmet.accuracy_score(lab_real, lab_pred)

    if (verbose):
        print('- Accuracy - {:.3f}'.format(acc))

    return acc


def topk_accuracy (lab_real, lab_pred, topk, verbose=False):
    """
    Computes the top k accuracy for the given data
    :param lab_real(np.array): the data real labels
    :param lab_pred(np.array): the predictions returned by the model
    :param topk (int): the top k labels to taking into account
    :param verbose(bool, optional): if you'd like to print the accuracy. Dafault is False.
    :return (float): the accuracy
    """
    
    # Checkin the array dimension
    _, lab_pred = _check_dim(lab_real, lab_pred, mode='scores')
    lab_pred = np.argsort(lab_pred, axis=1)[:,-topk:]

    n_samples = lab_real.shape[0]
    hits = 0

    for k in range(n_samples):
        if (lab_real[k] in lab_pred[k,:]):
            hits += 1

    acc = hits/n_samples
    if (verbose):
        print('- Top {} accuracy - {:.3f}'.format(topk, acc))

    return acc


def conf_matrix (lab_real, lab_pred, normalize=False):
    """
    This function computes the confusion matrix. Both lab_real and lab_pred can be a labels array or and a array of
    scores (one hot encoding) for each class.

    :param lab_real(np.array): the data real labels
    :param lab_pred(np.array): the predictions returned by the model
    :param class_names (list): the name of each label. For example: ['l1','l2']. If you pass a list with a different
    number of labels that provided in lad_pred or real, you're gonna have an exception. If None, the labels will not
    be considered. Default is None.
    :param normalize (bool, optional): set it True if you'd like to normalize the cm. Default is False.
    :return (2d np array: an np array containing the confusion matrix
    """

    # Checkin the array dimension
    lab_real, lab_pred = _check_dim(lab_real, lab_pred, mode='labels')

    cm = skmet.confusion_matrix(lab_real, lab_pred)

    if (normalize):
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    return cm


def plot_conf_matrix(cm, class_names, normalize=False, save_path=None, title='Confusion matrix', cmap=plt.cm.GnBu):
    """
    This function makes a plot for a given confusion matrix. It can plots either the real or normalized one.
    Most of this code is provided on:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

    Please, refers there for any further reference.

    :param cm (np.array): An m x m np array containing the confusion matrix
    :param class_names (list): a list with the class labels. Ex:['A', 'B'], if you have 2 labels
    :param normalize (bool, optional): set it True if you'd like to normalize the cm. Default is False.
    :param save_path (string, optional): if you'd like to save your plot instead of show it on the screen, you need to
    provide the full path (including image name and format) to do so. Ex: /home/user/cm.png. If None, the plot is not
    save but showed in the screen. Default is None.
    :param title (string, optional): the plot's title. Default is 'Confusion matrix'
    :param cmap (plt.cm.color, option): a color pallete provided by pyplot. Default is GnBu.
    """

    if (normalize):
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if (save_path is None):
        plt.show()
    else:
        plt.savefig(save_path)



def precision_recall_report (lab_real, lab_pred, class_names=None, verbose=False):
    """
    Computes the precision, recall, F1 score and support for each class. Both lab_real and lab_pred can be a labels
    array or and a array of scores (one hot encoding) for each class.

    :param lab_real (np.array): the data real labels
    :param lab_pred (np.array): the predictions returned by the model
    :param class_names (list): the name of each label. For example: ['l1','l2']. If you pass a list with a different
    number of labels that provided in lad_pred or real, you're gonna have an exception. If None, the labels will not
    be considered. Default is None.
    :return (string): a string containing the repost regarding each metric
    """

    # Checkin the array dimension
    lab_real, lab_pred = _check_dim(lab_real, lab_pred, mode='labels')

    report = skmet.classification_report(lab_real, lab_pred, target_names=class_names)

    if (verbose):
         print(report)

    return report


def auc_and_roc_curve (lab_real, lab_pred, class_names, class_to_compute='all', save_path=None):
    """
    This function computes the ROC curves and AUC for each class.
    It better described on: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py

    Both lab_real and lab_pred can be a labels array or and a array of scores (one hot encoding) for each class.

    :param lab_real (np.array): the data real labels
    :param lab_pred (np.array): the predictions returned by the model
    :param class_names (list): the name of each label. For example: ['l1','l2']. If you pass a list with a different
    :param class_to_compute (string, optional): select the class you'd like to compute the ROC. If you set 'all', it
    will compute all curves. Note that you should inform a valid class, that is, a class that is inside in class_name.
    Default is 'all'.
    :return: a dictionaty with the AUC, fpr, tpr for each class
    """

    # Checkin the array dimension
    lab_real, lab_pred = _check_dim(lab_real, lab_pred, mode='scores')

    # Computing the ROC curve and AUC for each class
    fpr = dict()  # false positive rate
    tpr = dict()  # true positive rate
    roc_auc = dict()  # area under the curve
    for i, name in enumerate(class_names):
        # print(i, name)
        fpr[name], tpr[name], _ = skmet.roc_curve(lab_real[:, i], lab_pred[:, i])
        roc_auc[name] = skmet.auc(fpr[name], tpr[name])


    if (class_to_compute == 'all'):

        # Computing the micro-average ROC curve and the AUC
        fpr["micro"], tpr["micro"], _ = skmet.roc_curve(lab_real.ravel(), lab_pred.ravel())
        roc_auc["micro"] = skmet.auc(fpr["micro"], tpr["micro"])

        # Ploting all ROC curves
        plt.figure()

        # Plotting the micro avg
        plt.plot(fpr["micro"], tpr["micro"],
                 label='ROC: AVG - AUC: {0:0.2f}'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=2)

        # Plottig the curves for each class
        for name in class_names:
            plt.plot(fpr[name], tpr[name], linewidth=1,
                     label='ROC: {0} - AUC: {1:0.2f}'
                           ''.format(name, roc_auc[name]))

    else:

        plt.plot(fpr[class_to_compute], tpr[class_to_compute], linewidth=1,
                 label='ROC: {0} - AUC: {1:0.2f}'
                       ''.format(class_to_compute, roc_auc[class_to_compute]))

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves')
    plt.legend(loc="lower right")
    
    if (save_path is None):
        plt.show()
    else:
        plt.savefig(save_path)
        plt.clf()

    return roc_auc, fpr, tpr