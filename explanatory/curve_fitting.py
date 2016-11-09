# -*- coding: utf-8 -*-
import sys,os
import math
import re
import argparse
import json
import ast
import copy
from subprocess import Popen, PIPE
from operator import itemgetter

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils/'))
from collection_stats import CollectionStats
from query import Query
from judgment import Judgment
from evaluation import Evaluation
from performance import Performances
from gen_doc_details import GenDocDetails

import numpy as np
import scipy.stats
from scipy.optimize import curve_fit

import unittest

class RealModels(object):
    """
    all kinds of real models
    """
    def __init__(self):
        super(RealModels, self).__init__()

    def okapi(self, collection_stats, row, k1=0.9, b=0.35):
        """
        okapi
        """
        return round((k1+1)*float(row['total_tf'])/(float(row['total_tf']) + k1*(1-b+b*float(row['doc_len'])/float(collection_stats.get_avdl()))), 3) 
    def tf1(self, collection_stats, row):
        """
        tf
        """
        return int(row['total_tf'])
    def tf4(self, collection_stats, row):
        """
        1+log(1+log(tf))
        """
        return round(1+math.log(1+math.log(int(row['total_tf']))), 3)
    def tf5(self, collection_stats, row):
        """
        tf/(tf+k)  k=1.0 default
        """
        return round(int(row['total_tf']) / (1.0 + int(row['total_tf'])), 4)
    def tfidf1(self, collection_stats, row):
        """
        tf/(tf+k) * idf  k=1.0 default
        """
        return round(int(row['total_tf']) / (1.0 + int(row['total_tf'])), 4)
    def tf_dl_1(self, collection_stats, row):
        """
        tf/dl
        """
        return round(float(row['total_tf'])/float(row['doc_len']), 3) 

    def tf_dl_3(self, collection_stats, row):
        """
        log(tf)/(tf+log(dl))
        """
        return round(np.log(float(row['total_tf']))/(float(row['total_tf'])+np.log(float(row['doc_len']))), 3) 
    def tf_dl_5(self, collection_stats, row, delta=2.75):
        """
        (log(tf)+delta)/(tf+log(dl))
        """
        return round((np.log(float(row['total_tf']))+delta)/np.log(float(row['doc_len'])), 3) 

    def get_func_mapping(self, method_name='tf1', para_str=''):
        formal_method_name = method_name
        if method_name == 'okapi':
            x_func = self.okapi
            formal_method_name = 'okapi,'+para_str #e.g. 'b:0.0'
        elif method_name == 'tf1':
            x_func = self.tf1
        elif method_name == 'tf4':
            x_func = self.tf4
        elif method_name == 'tf5':
            x_func = self.tf5
        elif method_name == 'tf_ln_1':
            x_func = self.tf_dl_1
            formal_method_name = 'hypothesis_stq_tf_ln_1'
        elif method_name == 'tf_ln_3':
            x_func = self.tf_dl_3
            formal_method_name = 'hypothesis_stq_tf_ln_3'
        elif method_name == 'tf_ln_5':
            x_func = self.tf_dl_5
            formal_method_name = 'hypothesis_stq_tf_ln_5'

        return x_func, formal_method_name


class FittingModels(object):
    """
    all kinds of fitting models
    """
    def __init__(self):
        super(FittingModels, self).__init__()

    def mix_expon1(self, xaxis, l):
        return scipy.stats.expon(scale=1.0/l).pdf(xaxis)
    def mix_expon2(self, xaxis, pi, l1, l2):
        return pi*scipy.stats.expon(scale=1.0/l1).pdf(xaxis) + (1-pi)*scipy.stats.expon(scale=1.0/l2).pdf(xaxis)
    def mix_expon3(self, xaxis, pi1, pi2, l1, l2, l3):
        return pi1*scipy.stats.expon(scale=1.0/l1).pdf(xaxis) + pi2*scipy.stats.expon(scale=1.0/l2).pdf(xaxis) + (1-pi1-pi2)*scipy.stats.expon(scale=1.0/l3).pdf(xaxis)
    def mix_expdecay1(self, xaxis, n0, l):
        return n0*np.exp(-l*xaxis)
    def mix_expdecay2(self, xaxis, pi, n01, n02, l1, l2):
        return pi*n01*np.exp(-l1*xaxis) + (1-pi)*n02*np.exp(-l2*xaxis)
    def asymptotic_decay(self, xaxis, n0, halflife):
        return n0*(1 - xaxis/(xaxis+halflife))
    def power_decay(self, xaxis, n0, halflife):
        return n0*np.power(xaxis, -halflife)

    def cal_curve_fit(self, xaxis, yaxis, mode=1, paras=[], bounds=(-np.inf, np.inf)):
        if mode == 1:
            func = self.mix_expon1
        elif mode == 2:
            func = self.mix_expon2
        elif mode == 3:
            func = self.mix_expon3
        elif mode == 4:
            func = self.mix_expdecay1
        elif mode == 5:
            func = self.mix_expdecay2
        elif mode == 6:
            func = self.asymptotic_decay
        elif mode == 7:
            func = self.power_decay
        xaxis = np.array(xaxis)
        try:
            popt, pcov = curve_fit(func, xaxis, yaxis, p0=paras, method='trf', bounds=bounds)
            perr = np.sqrt(np.diag(pcov))
            trialY = func(xaxis, *popt)
            print mode, popt, np.absolute(trialY-yaxis).sum(), scipy.stats.ks_2samp(yaxis, trialY)
        except:
            return None
        return popt, trialY, np.absolute(trialY-yaxis).sum(), scipy.stats.ks_2samp(yaxis, trialY)

class EM(object):
    """
    Expectation Maximization algorithm
    """

    def __init__(self):
        super(EM, self).__init__()

    def exponential(self, data=[], init_lambdas=[1,0.75], max_iteration=500):
        """
        two mixture of exponential
        """
        xaxis = np.arange(1, len(data)+1)
        data = np.array(data)
        idx = 1
        lambdas = np.array(init_lambdas)
        while idx < max_iteration:
            y = [lmbda*np.exp(data*(-lmbda)) for lmbda in lambdas]
            weights = y/np.sum(y, axis=0)
            coefficients = np.mean(weights, axis=1)
            lambdas = np.sum(weights, axis=1)/np.sum(weights*data, axis=1)
            idx+=1 
        print lambdas, coefficients
        return lambdas, coefficients

class Test(unittest.TestCase):
    pass
    

if __name__ == '__main__':
    #unittest.main()
    em = EM()
    a = [70,40,20,10,9,8,7,6,5,4,3,2,2,2,2,2,2,1,1,1,1,1]
    em.exponential(np.asarray(a)*1./np.sum(a))