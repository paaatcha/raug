#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: André Pacheco
E-mail: pacheco.comp@gmail.com

This file implements the metrics to be used in the evaluation

If you find any bug or have some suggestion, please, email me.
"""

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import itertools


def accuracy (outputs, labels, verbose=False):
    """
    Just a simple functin to compute the accuracy

    :param outputs: the predictions returned by the model
    :param labels: the data real labels
    :param verbose: if you'd like to print the accuracy. Dafault is False.
    :return: the accuracy
    """

    # correct = (outputs == labels).sum().item()
    # acc = correct / outputs.shape[0]
    acc = accuracy_score(labels, outputs)

    if (verbose):
        print('Accuracy - {:.3f}'.format(acc))

    return acc


def conf_matrix (outputs, labels, class_names=None):
    return confusion_matrix(labels, outputs, labels=class_names)

def plot_conf_matrix(cm, classes, normalize=False, save_path=None, title='Confusion matrix', cmap=plt.cm.GnBu):
    """
    This function makes a plot for a given confusion matrix. It can plots either the real or normalized one.
    Most of this code is provided on:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

    Please, refers there for any further reference.

    :param cm (np.array): An m x m np array containing the confusion matrix
    :param classes (list): a list with the class labels. Ex:['A', 'B'], if you have 2 labels
    :param normalize (bool, optional): set it True if you'd like to normalize the cm. Default is False.
    :param save_path (string, optional): if you'd like to save your plot instead of show it on the screen, you need to
    provide the full path (including image name and format) to do so. Ex: /home/user/cm.png. If None, the plot is not
    save but showed in the screen. Default is None.
    :param title (string, optional): the plot's title. Default is 'Confusion matrix'
    :param cmap (plt.cm.color, option): a color pallete provided by pyplot. Default is GnBu.
    """

    if (normalize):
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

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


def precision_recall_report (outputs, labels, verbose=False):
    """
    Just a simple functin to compute the accuracy

    :param outputs: the predictions returned by the model
    :param labels: the data real labels
    :param verbose: if you'd like to print the accuracy. Dafault is False.
    :return: the accuracy
    """

    # correct = (outputs == labels).sum().item()
    # acc = correct / outputs.shape[0]
    acc = classification_report(labels, outputs, target_names = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6'])

    print (acc)

    # if (verbose):
    #     print('Precision - {:.3f}'.format(acc))

    return acc