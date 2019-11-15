#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: André Pacheco
E-mail: pacheco.comp@gmail.com

This file contains my implementations regarding the outlier and ensemble weights

If you find any bug or have some suggestion, please, email me.
"""

import sys
sys.path.insert(0,'../../dirichlet') # Adding the dirichlet path
from dirichlet import dirichlet
import numpy as np
import os
import pandas as pd
import glob
from .common import insert_pred_col
from scipy.special import softmax
from ..pytorch.model.metrics import AVGMetrics
from scipy.spatial.distance import cosine, euclidean, jensenshannon, mahalanobis, correlation
from scipy.stats import wasserstein_distance, energy_distance

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

def metric (x, y, name, alphas=None, ep=0.000001):
    if name == "kld_div":
        return kld_div(y, x, ep)
    elif name == "euclidean":
        return euclidean(x,y)
    elif name == "wasserstein":
        return wasserstein_distance(x, y)
    elif name == "hellinger":
        return hellinger_explicit(x, y)
    elif name == "bhattcharyya":
        return bha_dis(x,y)
    elif name == "jensen-shannon":
        return jensenshannon(x, y)
    elif name == "correlation":
        return correlation(x, y)
    elif name == "cosine":
        return cosine(x, y)
    elif name == "energy":
        return energy_distance(x, y)
    elif name == "mahalanobis":
        if alphas is None:
            print ("You need to pass alpha to compute mahalanobis")
            raise ValueError
        else:
            return maha_dist(x, y, alphas)
    else:
        print("There is no distance called", type)
        raise ValueError

def kld_div (PY, PX, ep=0.000001):
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

def bha_dis (PX, PY, ep=0.000001):
    """ Bhattacharyya distance """
    PX = np.asarray(PX, np.float)
    PY = np.asarray(PY, np.float)
    PX = PX + ep
    PY = PY + ep
    bc = np.sqrt(PX * PY).sum()
    return -np.log(bc)

def hellinger_explicit(p, q):
    """Hellinger distance between two discrete distributions.
       Same as original version but without list comprehension
    """
    list_of_squares = []
    for p_i, q_i in zip(p, q):

        # caluclate the square of the difference of ith distr elements
        s = (np.sqrt(p_i) - np.sqrt(q_i)) ** 2

        # append
        list_of_squares.append(s)

    # calculate sum of squares
    sosq = sum(list_of_squares)

    return sosq / np.sqrt(2)

def dirichlet_cov_var (alphas):
    n = len(alphas)
    cov = np.zeros((n,n))
    a0 = sum(alphas)
    for i in range(n):
        for j in range(n):
            if i == j:
                cov[i,j] = (alphas[i]*(a0 - alphas[i])) / ( (a0**2) * (a0 + 1) )
            else:
                cov[i,j] = -(alphas[i] * alphas[j]) / ((a0**2) * (a0 + 1))
    return cov

def maha_dist (pi, pj, alphas):
    # if cov is None:
    #     # print ("Computing cov...")
    #     # cov = np.cov(np.stack((pi, pj), axis=1))
    cov = dirichlet_cov_var (alphas)
    return mahalanobis(pi, pj, np.linalg.pinv(cov))


def models_weights (hit, miss):

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

def compute_logit_stats (data, labels_name, col_pred="PRED", pred_pos=2, col_true="REAL", dir_met="fixedpoint",
                         max_iter=5000, tol=1e-5):
    
    print ("-"*50)
    print("- Starting the logit stats computation...")
    # If the data is a path, we load it.
    if isinstance(data, str):
        print("- Loading data...")
        output_path = data
        data = pd.read_csv(data)


    # Checking if we need to include the prediction column or the DataFrame already has it.
    data = insert_pred_col(data, labels_name, pred_pos=pred_pos, col_pred=col_pred)

    # Computing the entropy and inserting as a column in the DataFrame
    # print("- Computing the entropy for all samples...")
    # data['Entropy'] = data[labels_name].apply(entropy, axis=1)

    # Dict to save the stats
    logit_stats = dict()

    # Now we're going to compute the max, min and avg entropy for each label and considering the hits and misses:    
    for lab in labels_name:
        print("- Computing the stats for {}...".format(lab))
        d_lab = data[data[col_true] == lab]
        d_hit = d_lab[d_lab[col_true] == d_lab[col_pred]]
        d_miss = d_lab[d_lab[col_true] != d_lab[col_pred]]

        try:
            Dlab = d_lab[labels_name].values
            alphas_lab = dirichlet.mle(Dlab, method=dir_met, tol=tol, maxiter=max_iter)
        except Exception:
            print("Dirichlet did not converged to {} all labels".format(lab))
            alphas_lab = None

        try:
            Dhit = d_hit[labels_name].values
            alphas_hit = dirichlet.mle(Dhit, method=dir_met, tol=tol, maxiter=max_iter)
        except Exception:
            print("Dirichlet did not converged to {} hit labels".format(lab))
            alphas_hit = None

        try:
            Dmiss = d_miss[labels_name].values
            alphas_miss = dirichlet.mle(Dmiss, method=dir_met, tol=tol, maxiter=max_iter)
        except Exception:
            print("Dirichlet did not converged to {} miss labels".format(lab))
            alphas_miss = None

        logit_stats[lab] = {
            'hit': {
                # 'max_ent': d_hit['Entropy'].max(),
                # 'min_ent': d_hit['Entropy'].min(),
                # 'avg_ent': d_hit['Entropy'].mean(),
                # 'std_ent': d_hit['Entropy'].std(),
                'avg_prob': d_hit[labels_name].mean(),
                'std_prob': d_hit[labels_name].std(),
                'alphas': alphas_hit

            },
            'miss': {
                # 'max_ent': d_miss['Entropy'].max(),
                # 'min_ent': d_miss['Entropy'].min(),
                # 'avg_ent': d_miss['Entropy'].mean(),
                # 'std_ent': d_miss['Entropy'].std(),
                'avg_prob': d_miss[labels_name].mean(),
                'std_prob': d_miss[labels_name].std(),
                'alphas': alphas_miss
            },
            'all': {
                # 'avg_ent': d_lab['Entropy'].mean(),
                # 'std_ent': d_lab['Entropy'].std(),
                'avg_prob': d_lab[labels_name].mean(),
                'std_prob': d_lab[labels_name].std(),
                'alphas': alphas_lab
            }
        }

    return logit_stats