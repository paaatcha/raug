#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: André Pacheco
E-mail: pacheco.comp@gmail.com

This file implements some functions to perform classification metrics

If you find any bug or have some suggestion, please, email me.
"""

import sklearn.metrics as skmet
import matplotlib.pyplot as plt
import numpy as np
import itertools
import pandas as pd

def _one_hot_encoding(ind, N=None):
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
    if mode == 'labels':
        if lab_real.ndim == 2:
            lab_real = lab_real.argmax(axis=1)
        if lab_pred.ndim == 2:
            lab_pred = lab_pred.argmax(axis=1)

    elif mode == 'scores':
        if lab_real.ndim == 1:
            lab_real = _one_hot_encoding(lab_real)
        if lab_pred.ndim == 1:
            lab_pred = _one_hot_encoding(lab_pred)

    else:
        raise Exception ('There is no mode called {}. Please, choose between score or labels'.format(mode))

    return lab_real, lab_pred


class AVGMetrics (object):
    """
        This is a simple class to control the AVG for a given value. It's used to control loss and accuracy for start
        and evaluate partition
    """
    def __init__(self):
        self.sum_value = 0
        self.avg = 0
        self.count = 0
        self.values = []

    def __call__(self):
        return self.avg

    def update(self, val):
        self.values.append(val)
        self.sum_value += val
        self.count += 1
        self.avg = self.sum_value / float(self.count)

    def print (self):
        print('\nsum_value: ', self.sum_value)
        print('count: ', self.count)
        print('avg: ', self.avg)


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

    if verbose:
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
        if lab_real[k] in lab_pred[k,:]:
            hits += 1

    acc = hits/n_samples
    if verbose:
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

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    return cm


def plot_conf_matrix(cm, class_names, normalize=True, save_path=None, title='Confusion matrix', cmap=plt.cm.GnBu):
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

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=0)
    plt.yticks(tick_marks, class_names)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    if isinstance(save_path, str):
        plt.savefig(save_path, dpi=200)
        plt.clf()
    elif save_path:
        plt.show()
    else:
        plt.clf()


def roc_auc (lab_real, lab_pred):
    """
    This computes the ROC AUC for binary classification tasks.

    :param lab_real (np.array): the data real labels
    :param lab_pred (np.array): the predictions returned by the model
    :return (number): the AUC
    """

    lab_real, lab_pred = _check_dim(lab_real, lab_pred, mode='scores')
    return skmet.roc_auc_score(lab_real, lab_pred)


def balanced_accuracy (lab_real, lab_pred):
    """
    This computes the balance accuracy for binary or multiclass classification tasks. This metric is the average recall
    or sensitivity

    :param lab_real (np.array): the data real labels
    :param lab_pred (np.array): the predictions returned by the model
    :return (number): the balanced accuracy
    """

    lab_real, lab_pred = _check_dim(lab_real, lab_pred, mode='labels')
    return skmet.balanced_accuracy_score(lab_real, lab_pred)


def precision_recall_report (lab_real, lab_pred, class_names=None, verbose=False, output_dict=False):
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

    # Checking the array dimension
    lab_real, lab_pred = _check_dim(lab_real, lab_pred, mode='labels')

    report = skmet.classification_report(lab_real, lab_pred, target_names=class_names, output_dict=output_dict)

    if verbose:
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

    if class_to_compute == 'all':

        # Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[name] for name in class_names]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for name in class_names:
            mean_tpr += np.interp(all_fpr, fpr[name], tpr[name])

        # Finally average it and compute AUC
        mean_tpr /= float(len(class_names))

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = skmet.auc(fpr["macro"], tpr["macro"])

        # Computing the micro-average ROC curve and the AUC
        fpr["micro"], tpr["micro"], _ = skmet.roc_curve(lab_real.ravel(), lab_pred.ravel())
        roc_auc["micro"] = skmet.auc(fpr["micro"], tpr["micro"])

        if save_path:
            # Ploting all ROC curves
            plt.figure()

            # Plotting the micro avg
            plt.plot(fpr["micro"], tpr["micro"],
                     label='MicroAVG - AUC: {0:0.2f}'
                           ''.format(roc_auc["micro"]),
                     color='deeppink', linestyle=':', linewidth=2)

            # Plotting the micro avg
            plt.plot(fpr["macro"], tpr["macro"],
                     label='MacroAVG - AUC: {0:0.2f}'
                           ''.format(roc_auc["macro"]),
                     color='navy', linestyle=':', linewidth=2)

            # Plottig the curves for each class
            for name in class_names:
                plt.plot(fpr[name], tpr[name], linewidth=1,
                         label='{0} - AUC: {1:0.2f}'
                               ''.format(name, roc_auc[name]))

    else:

        if save_path:
            plt.plot(fpr[class_to_compute], tpr[class_to_compute], linewidth=1,
                     label='{0} - AUC: {1:0.2f}'
                           ''.format(class_to_compute, roc_auc[class_to_compute]))

    if save_path:
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curves')
        plt.legend(loc="lower right")

        if isinstance(save_path, str):
            plt.savefig(save_path)
            plt.clf()
        elif save_path:
            plt.show()


    return roc_auc, fpr, tpr


def get_metrics_from_csv (csv, class_names=None, topk=2, conf_mat=False, conf_mat_path=None, roc=False,
                          roc_path=None, verbose=True):

    if isinstance(csv, str):
        data = pd.read_csv(csv)
    else:
        data = csv

    if class_names is None:
        class_names = data.columns.values[1:]

    class_names_dict = {name: pos for pos, name in enumerate(class_names)}
    preds = data[class_names].values

    try:
        labels_str = data['REAL'].values
    except KeyError:
        print ("Warning: There is no ground truth in this file! The code will return None")
        return None

    labels = [class_names_dict[lstr] for lstr in labels_str]
    labels = np.array(labels)

    acc = accuracy(labels, preds)
    topk_acc = topk_accuracy(labels, preds, topk)
    ba = balanced_accuracy(labels, preds)
    rep =  precision_recall_report(labels, preds, class_names, output_dict=True)
    loss = skmet.log_loss(labels, preds)

    if conf_mat:
        plt.figure()
        cm = conf_matrix(labels, preds, normalize=True)
        if conf_mat_path is None:
            p = "./conf.png"
        else:
            p = conf_mat_path
        plot_conf_matrix(cm, class_names, title='Confusion matrix', cmap=plt.cm.GnBu, save_path=p)

    if roc:
        plt.figure()
        auc, fpr, tpr = auc_and_roc_curve(labels, preds, class_names, save_path=roc_path)
    else:
        auc, fpr, tpr = auc_and_roc_curve(labels, preds, class_names, save_path=None)


    if verbose:
        print("-" * 50)
        print("- Metrics:")
        print("- Loss: {:.3f}".format(loss))
        print("- Accuracy: {:.3f}".format(acc))
        print("- Top {} Accuracy: {:.3f}".format(topk, topk_acc))
        print("- Balanced accuracy: {:.3f}".format(ba))
        print("- AUC macro: {:.3f}".format(auc['macro']))

    return acc, topk_acc, ba, rep, auc, loss, fpr, tpr

