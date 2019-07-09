#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: André Pacheco
E-mail: pacheco.comp@gmail.com

This file implements the class metrics to be used in the evaluation phase

If you find any bug or have some suggestion, please, email me.
"""

import os
import torch
import numpy as np
from ...utils import classification_metrics as cmet
import pandas as pd


class Metrics:

    def __init__(self, metrics_names, class_names=None, options=None):
        """
        Construction method. You must inform the metrics you'd like to compute. Optionally, you may set some options 
        and the classes names. Each parameter is better described bellow:
        
        :param metrics_name (list, tuple or string): it's the variable that receives the metrics you'd like to compute.
        The following metrics are available: "accuracy", "topk_accuracy", "balanced_accuracy", "conf_matrix", 
        "plot_conf_matrix", "precision_recall_report", and "auc_and_roc_curve". To understand them, please, go to 
        jedy.utils.classification_metrics.py. You should pass one or more of these metrics in a list or a tuple, 
        for ex: m = ["accuracy", "conf_matrix"]. If you'd like to compute all of them, just set it as 'all', i.e., 
        m = 'all'
        
        Important: if you'd like to compute either 'plot_conf_matrix' or 'auc_and_roc_curve', you must inform the
        class_names. If not, you'll get an exception. The remaining metrics, except 'accuracy', also use the class_name,
        however, it's not demanded for them
        
        :param class_names (list, tuple): a list or tuple containing the classes names in the same order you use in the
        label. For ex: ['C1', 'C2']

        :param options (dict): this is a dict containing some options to compute the metrics. The following options are
        available:
        - For all:
            - save_all_path: a string with the path to save all metrics and images. In this case, the conf matrix will be
            called by conf_mat.png and the roc curve as roc_curve.png

        - For 'topk_accuracy':
            - 'topk' (int): if you'd like to compute the top k accuracy, you should inform the top k. If you don't,
            the default value is 2

        - For 'conf_matrix': 
            - 'normalize_conf_matrix' (bool): inform if you'd like to normalize the confusion matrix
        
        - For 'plot_conf_matrix':
            - 'save_path_conf_matrix' (string): the complete file path if you'd like to save instead show the plot
            - 'normalize_conf_matrix' (bool): inform if you'd like to normalize the confusion matrix
            - 'title_conf_matrix' (string): the plot's title
            
        - For 'auc_and_roc_curve':
            - 'save_path_roc_curve' (string): the complete file path if you'd like to save instead show the plot 
            - 'class_to_compute_roc_curve' (string): if you'd like to compute just one class instead all, you can set it
            here.

        - For 'save_scores':
            - 'save_path_scores' (string): the complete file path if to save the scores
            - 'pred_name_scores' (string): the names of the predictions .csv. If it's not informed, it's going to be
            predictions.csv
            
        For more information about the options, please, refers to jedy.utils.classification_metrics.py
         
        """
        self.metrics_names = metrics_names
        self.metrics_values = dict()        
        self.options = options
        
        self.pred_scores = None
        self.label_scores = None
        
        self.class_names = class_names
        self.topk = None




    def compute_metrics (self):
        """
        This method computes all metrics defined in metrics_name.
        :return: it saves in self.metric_values all computed metrics
        """
        
        if self.metrics_names is None:
            return None

        save_all_path = None
        # Checking if save_all is informed
        if (self.options is not None):
            if ("save_all_path" in self.options.keys()):
                save_all_path = self.options["save_all_path"]
        
        if (self.metrics_names == "all"):
            self.metrics_names = ["accuracy", "topk_accuracy", "balanced_accuracy",  "conf_matrix", "plot_conf_matrix",
                                  "precision_recall_report", "auc_and_roc_curve"]
        
        
        for mets in self.metrics_names:
            if (mets == "accuracy"):
                self.metrics_values["accuracy"] = cmet.accuracy(self.label_scores, self.pred_scores)
                
            elif (mets == "balanced_accuracy"):
                self.metrics_values["balanced_accuracy"] = cmet.balanced_accuracy(self.label_scores, self.pred_scores)

            elif (mets == "topk_accuracy"):

                # Checking if the class names are defined
                self.topk = 2
                if (self.options is not None):
                    if ("topk" in self.options.keys()):
                        self.topk = self.options["topk"]

                self.metrics_values["topk_accuracy"] = cmet.topk_accuracy(self.label_scores, self.pred_scores, self.topk)
            
            elif (mets == "conf_matrix"):
                
                # Checking the options
                normalize = False
                if (self.options is not None):
                    if ("normalize_conf_matrix" in self.options.keys()):
                        normalize = self.options["normalize_conf_matrix"]
                
                self.metrics_values["conf_matrix"] = cmet.conf_matrix(self.label_scores, self.pred_scores, normalize)
            elif (mets == "plot_conf_matrix"):
                
                # Checking if the class names are defined
                if (self.class_names is None):
                    raise Exception ("You are trying to plot the confusion matrix without defining the classes name")
                
                # Checking the options
                save_path = None
                normalize = False
                title = "Confusion Matrix"   
                
                if (self.options is not None):
                    if (save_all_path is not None):
                        save_path = os.path.join(save_all_path, "conf_mat.png")
                    if ("save_path_conf_matrix" in self.options.keys()):
                        save_path = self.options["save_path_conf_matrix"]
                    if ("normalize_conf_matrix" in self.options.keys()):
                        normalize = self.options["normalize_conf_matrix"]
                    if ("title_conf_matrix" in self.options.keys()):
                        title = self.options["title_conf_matrix"]
                        
                if ("conf_matrix" in self.metrics_values.keys()):
                    cm = self.metrics_values["conf_matrix"]
                else:
                    cm = cmet.conf_matrix(self.label_scores, self.pred_scores, normalize)
                
                cmet.plot_conf_matrix(cm, self.class_names, normalize, save_path, title)
                
                
            elif (mets == "precision_recall_report"):

                self.metrics_values["precision_recall_report"] = cmet.precision_recall_report(self.label_scores,
                                                                                              self.pred_scores,
                                                                                              self.class_names)
                
            elif (mets == "auc_and_roc_curve"):
                
                # Checking if the class names are defined
                if (self.class_names is None):
                    raise Exception ("You are trying to plot the confusion matrix without defining the classes name")

                # Checking the options
                save_path = None
                class_to_compute = "all"                

                if (self.options is not None):
                    if (save_all_path is not None):
                        save_path = os.path.join(save_all_path, "roc_curve.png")
                    if ("save_path_roc_curve" in self.options.keys()):
                        save_path = self.options["save_path_roc_curve"]
                    if ("class_to_compute_roc_curve" in self.options.keys()):
                        class_to_compute = self.options["class_to_compute_roc_curve"]

                self.metrics_values["auc_and_roc_curve"] = cmet.auc_and_roc_curve(self.label_scores, self.pred_scores,
                                                                                  self.class_names, class_to_compute, 
                                                                                  save_path)

            if (save_all_path is not None):
                self.save_metrics(save_all_path)

    def print (self):
        """
        This method just prints the metrics on the screen
        """
        
        if self.metrics_names is None:
            print ("Since metrics name is None, there is no metric to print")
            
        else:        
            for met in self.metrics_values.keys():
                if (met == "loss"):
                    print ('- Loss: {:.3f}'.format(self.metrics_values[met]))
                elif (met == "accuracy"):
                    print ('- Accuracy: {:.3f}'.format(self.metrics_values[met]))
                elif (met == "balanced_accuracy"):
                    print ('- Balanced accuracy: {:.3f}'.format(self.metrics_values[met]))
                elif (met == "topk_accuracy"):
                    print('- Top {} accuracy: {:.3f}'.format(self.topk, self.metrics_values[met]))
                elif (met == "conf_matrix"):
                    print('- Confusion Matrix: \n{}'.format(self.metrics_values[met]))
                elif (met == "precision_recall_report"):
                    print('- Precision and Recall report: \n{}'.format(self.metrics_values[met]))
                elif (met == "auc_and_roc_curve"):
                    resp = self.metrics_values[met]
                    print('- AUC: \n{}'.format(resp[0]))

    def add_metric_value (self, value_name, value):
        """
        Adding a new value from a external source into the metrics
        :param value_name (string): the key for the dict
        :param value: the value to be saved in the self.metrics_values
        """
        self.metrics_values[value_name] = value


    def update_scores (self, label_batch, pred_batch):
        """
        The evaluation is made using batchs. So, every batch we get just a piece of the prediction. This method
        concatenate all prediction and labels in order to compute the metrics
        :param pred (np.array): an array containing part of the predictions outputed by the model
        :param label (np.array): an array contaning the true labels
        """

        if (self.label_scores is None and self.pred_scores is None):
            self.label_scores = label_batch
            self.pred_scores = pred_batch
        else:
            self.pred_scores = np.concatenate((self.pred_scores, pred_batch))
            self.label_scores = np.concatenate((self.label_scores, label_batch))


    def save_metrics (self, folder_path, name="metrics.txt"):
        """
        This method saves the computed metrics
        :param folder_path (string): the folder you'd like to save the metrics
        :param name (string): the file name. Default is metrics.txt
        """
        
        if self.metrics_names is None:
            print ("Since metrics name is None, there is no metric to save")
            
        else:        
            with open(os.path.join(folder_path, name), "w") as f:
    
                f.write("- METRICS REPORT -\n\n")
    
                for met in self.metrics_values.keys():
                    if (met == "loss"):
                        f.write('- Loss: {:.3f}\n'.format(self.metrics_values[met]))
                    elif (met == "accuracy"):
                        f.write('- Accuracy: {:.3f}\n'.format(self.metrics_values[met]))
                    elif (met == "balanced_accuracy"):
                        f.write('- Balanced accuracy: {:.3f}\n'.format(self.metrics_values[met]))
                    elif (met == "topk_accuracy"):
                        f.write('- Top {} accuracy: {:.3f}\n'.format(self.topk, self.metrics_values[met]))
                    elif (met == "conf_matrix"):
                        f.write('- Confusion Matrix: \n{}\n'.format(self.metrics_values[met]))
                    elif (met == "precision_recall_report"):
                        f.write('- Precision and Recall report: \n{}\n'.format(self.metrics_values[met]))
                    elif (met == "auc_and_roc_curve"):
                        resp = self.metrics_values[met]
                        f.write('- AUC:\n {}'.format(resp[0]))


    def save_scores (self, folder_path=None, pred_name="predictions.csv"):
        """
        This method saves the concatenated scores in the disk
        :param folder_path (string): the folder you'd like to save the scores
        :param pred_name (string): the predictions' score file name. Default is predictions.csv
        :param labels_name (string): the labels' score file name. Default is labels.csv
        """

        if folder_path is not None:
            # Checking if the folder doesn't exist. If True, we must create it.
            if (not os.path.isdir(folder_path)):
                os.mkdir(folder_path)
        elif self.options is not None:
            if "save_all_path" in self.options.keys():
                folder_path = self.options["save_all_path"]
            elif "save_path_scores" in self.options.keys():
                folder_path = self.options["save_path_scores"]
            else:
                raise ("The options doesnt have any folder to save the scores")

            if 'pred_name_scores' in self.options.keys():
                pred_name = self.options['pred_name_scores']

        else:
            raise ("You must set the path to save the score eithe in options or in folder_path parameter")


        # Getting the list of classications and predict labels
        if self.class_names is not None:
            real_labels = [self.class_names[int(l)] for l in self.label_scores]
            pred_labels = [self.class_names[ps.argmax()] for ps in self.pred_scores]
        else:
            raise ("You need to inform the class names to use this function")

        real_labels = np.asarray(real_labels)
        pred_labels = np.asarray(pred_labels)
        real_labels = real_labels.reshape(real_labels.shape[0], 1)
        pred_labels = pred_labels.reshape(pred_labels.shape[0], 1)

        both_data = np.concatenate((real_labels, pred_labels, self.pred_scores), axis=1)

        cols = ['REAL', 'PRED', *self.class_names]
        df = pd.DataFrame(both_data, columns=cols)
        print ("Saving the scores in {}".format(folder_path))

        df.to_csv(os.path.join(folder_path, pred_name))


def accuracy (output, target, topk=(1,)):
    """
    This function computes the accuracy and top k accuracy for a given predictions and targets
    The difference between this and that one computed by the class is that this one is faster since it is made on
    GPU and during the training/validations phase. There is no need to wait for all batches release their scores
    such as in the classes. However, it's impossible to compute some metrics such as confusion matrix and AUC using
    this function.

    :param output: the predictions outputed by the model
    :param target: the ground truth
    :param topk: the top k accuracy, for example, top 5
    :return: the accuracy an topk accuracy
    """

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


class AVGMetrics (object):
    """
        This is a simple class to control the AVG for a given value. It's used to control loss and accuracy for train
        and evaluate partition
    """
    def __init__(self):
        self.sum_value = 0
        self.avg = 0
        self.count = 0

    def __call__(self):
        return self.avg

    def update(self, val):
        self.sum_value += val
        self.count += 1
        self.avg = self.sum_value / float(self.count)

    def print (self):
        print('\nsum_value: ', self.sum_value)
        print('count: ', self.count)
        print('avg: ', self.avg)
