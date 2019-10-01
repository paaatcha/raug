#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: André Pacheco
E-mail: pacheco.comp@gmail.com

This file contains my implementations regarding the outlier and ensemble weights

If you find any bug or have some suggestion, please, email me.
"""

import numpy as np
import os
import pandas as pd
import glob
from .common import insert_pred_col
from scipy.special import softmax
from ..pytorch.model.metrics import AVGMetrics

def entropy(x, ep=0.00001):
    """
    This function computes the Shannon entropy of a probability set
    :param x (numpy array): an array containing the probabilities
    :param ep (float): an very small value to avoid log(0)
    :return: the shannon entropy
    """
    ents = -np.sum(x * np.log2(x+ep))
    return ents

def tp (x, a):
    return np.power(x,a) / (np.power(x,a) + np.power((1 - x), a))

def KLD (PX, PY, ep=0.0001):
    """
    This functino computes the Kulback Leibler divergence between two set of probabilities
    :param PX (numpy array): an array containing the probabilities
    :param PY (numpy array): an array containing the probabilities
    :param ep (float): an very small value to avoid log(0)
    :return: the KL divergence between PX and PY
    """
    PX = np.asarray(PX, np.float)
    PY = np.asarray(PY, np.float)
    PX = PX + ep
    PY = PY + ep
    return (PX*np.log(PX/PY)).sum()


def compute_avg_KLD (stats, data, labels):
    avg_kld_hit = AVGMetrics()
    avg_kld_miss = AVGMetrics()
    for idx, sample in data.iterrows():

        samp_pred = sample['PRED']
        samp_agg_probs = sample[labels]

        kld_hit = KLD(samp_agg_probs.values, stats[samp_pred]['hit']['avg_prob'].values)
        kld_miss = KLD(samp_agg_probs.values, stats[samp_pred]['miss']['avg_prob'].values)

        avg_kld_miss.update(kld_miss)
        avg_kld_hit.update(kld_hit)

    return avg_kld_hit(), avg_kld_miss()

def KLD_weights (hit, miss):

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    h = np.array(hit)
    m = np.array(miss)
    s = m/h
#    s = s/s.max()
#    s = (s - s.min()) / (s.max() - s.min())
#     return softmax(s)
#    return sigmoid(s)
    return s / s.sum()

def compute_logit_stats (data, labels_name, col_pred="PRED", pred_pos=2, col_true="REAL"):
    
    print ("-"*50)
    print("- Starting the logit stats computation...")
    # If the data is a path, we load it.
    if isinstance(data, str):
        print("- Loading data...")
        data = pd.read_csv(data)

    # Checking if we need to include the prediction column or the DataFrame already has it.
    data = insert_pred_col(data, labels_name, pred_pos=pred_pos, col_pred=col_pred)

    # Computing the entropy and inserting as a column in the DataFrame
    print("- Computing the entropy for all samples...")
    data['Entropy'] = data[labels_name].apply(entropy, axis=1)

    # Dict to save the stats
    logit_stats = dict()

    # Now we're going to compute the max, min and avg entropy for each label and considering the hits and misses:    
    for lab in labels_name:
        print("- Computing the stats for {}...".format(lab))
        d_lab = data[data[col_true] == lab]
        d_hit = d_lab[d_lab[col_true] == d_lab[col_pred]]
        d_miss = d_lab[d_lab[col_true] != d_lab[col_pred]]

        logit_stats[lab] = {
            'hit': {
                'max_ent': d_hit['Entropy'].max(),
                'min_ent': d_hit['Entropy'].min(),
                'avg_ent': d_hit['Entropy'].mean(),
                'std_ent': d_hit['Entropy'].std(),
                'avg_prob': d_hit[labels_name].mean(),
                'std_prob': d_hit[labels_name].std()
            },
            'miss': {
                'max_ent': d_miss['Entropy'].max(),
                'min_ent': d_miss['Entropy'].min(),
                'avg_ent': d_miss['Entropy'].mean(),
                'std_ent': d_miss['Entropy'].std(),
                'avg_prob': d_miss[labels_name].mean(),
                'std_prob': d_miss[labels_name].std()
            },
            'all': {
                'avg_ent': d_lab['Entropy'].mean(),
                'std_ent': d_lab['Entropy'].std(),
                'avg_prob': d_lab[labels_name].mean(),
                'std_prob': d_lab[labels_name].std()
            }
        }

    return logit_stats