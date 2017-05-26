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
from abc import ABCMeta,abstractmethod

from emap import EMAP

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

    def okapi_apply(self, tf, idf, doclen, avdl, k1=1.2, b=0.35):
        return (k1+1)*idf*tf/(tf+k1*(1-b+b*doclen/avdl))
    def okapi(self, collection_stats, tf, df, doclen, k1=1.2, b=0.35):
        """
        okapi
        """
        idfs = np.log((collection_stats.get_doc_counts() + 1)/(df+1e-4))
        avdl = collection_stats.get_avdl()
        r = (k1+1)*idf*tf/(tf+k1*(1-b+b*doclen/avdl))
        # r = np.apply_along_axis(self.okapi_apply, 0, tf, idfs, doclen, avdl, k1, b)
        return np.sum(r, axis=0)
    def tf1(self, collection_stats, tf, df, doclen):
        """
        tf - numpy matrix (even if there is only one term), each row is the tf values for each term
        doclen - numpy array
        """
        return np.sum(tf, axis=0)
    def tf4(self, collection_stats, row):
        """
        1+log(1+log(tf))
        """
        return round(1+math.log(1+math.log(int(row['total_tf']))), 3)
    def tf5_apply(self, tf, k):
        return tf/(tf+k)
    def tf5(self, collection_stats, tf, df, doclen):
        """
        tf/(tf+k)  k=1.0 default
        """
        k = 1.0
        r = np.apply_along_axis(self.tf5_apply, 0, tf, [k])
        return np.sum(r, axis=0)
    def tfidf1_apply(self, tf, idf):
        return idf*tf/(tf+1)
    def tfidf1(self, collection_stats, tf, df, doclen):
        """
        tf/(tf+k) * idf  k=1.0 default
        """
        idfs = np.log((collection_stats.get_doc_counts() + 1)/(df+1e-4))
        r = np.apply_along_axis(self.tfidf1_apply, 0, tf, idfs)
        return np.sum(r, axis=0)
    def tfln1_apply(self, tf, doclen):
        return tf/doclen
    def tfln1(self, collection_stats, tf, df, doclen):
        """
        tf/dl
        """
        r = np.apply_along_axis(self.tfln1_apply, 0, tf, doclen)
        return np.sum(r, axis=0)

    def tfln3(self, collection_stats, row):
        """
        log(tf)/(tf+log(dl))
        """
        return round(np.log(float(row['total_tf']))/(float(row['total_tf'])+np.log(float(row['doc_len']))), 3) 
    def tfln5(self, collection_stats, row, delta=2.75):
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
        elif method_name == 'tfln1':
            x_func = self.tfln1
        elif method_name == 'tfln3':
            x_func = self.tfln3
        elif method_name == 'tfln5':
            x_func = self.tfln5
        elif method_name == 'tfidf1':
            x_func = self.tfidf1

        return x_func, formal_method_name


class FittingModels(object):
    """
    all kinds of fitting models
    """
    def __init__(self):
        super(FittingModels, self).__init__()

    def size(self):
        return 14

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
    def mix_lognormal1(self, xaxis, sigma):
        return scipy.stats.lognorm.pdf(xaxis, sigma)
    def mix_lognormal2(self, xaxis, pi, sigma1, sigma2):
        return pi*scipy.stats.lognorm.pdf(xaxis, sigma1)+(1-pi)*scipy.stats.lognorm.pdf(xaxis, sigma2)
    def mix_normal1(self, xaxis, mu, sigma):
        return scipy.stats.norm.pdf(xaxis, loc=mu, scale=sigma)
    def mix_normal2(self, xaxis, pi, mu1, mu2, sigma1, sigma2):
        return pi*scipy.stats.norm.pdf(xaxis, loc=mu1, scale=sigma1)+(1-pi)*scipy.stats.norm.pdf(xaxis, loc=mu2, scale=sigma2)
    def mix_gamma1(self, xaxis, a):
        return scipy.stats.gamma.pdf(xaxis, a)
    def mix_gamma2(self, xaxis, pi, a1, a2):
        return pi*scipy.stats.gamma.pdf(xaxis, a1)+(1-pi)*scipy.stats.gamma.pdf(xaxis, a2)
    def mix_poisson1(self, xaxis, mu):
        return scipy.stats.poisson.pmf(xaxis, mu)
    def mix_poisson2(self, xaxis, pi, mu1, mu2):
        return pi*scipy.stats.poisson.pmf(xaxis, mu1)+(1-pi)*scipy.stats.poisson.pmf(xaxis, mu2)

    def curve_fit_mapping(self, i):
        fitting_list = [self.mix_expon1, self.mix_expon2, self.mix_lognormal1, 
            self.mix_lognormal2, self.mix_normal1, self.mix_normal2, self.mix_gamma1, 
            self.mix_gamma2, self.mix_poisson1, self.mix_poisson2, self.asymptotic_decay, 
            self.power_decay, self.mix_expdecay1, self.mix_expdecay2] 
        return fitting_list[i-1]

    def cal_curve_fit(self, xaxis, yaxis, mode=1):
        if mode == 1:
            p0 = [1]
            bounds = ([0], [np.inf])
            func_name = 'EXP'
        elif mode == 2:
            p0 = [0.5, 2, 0.5]
            bounds = ([0, 0, 0], [1, np.inf, np.inf])
            func_name = '2-EXP'
        elif mode == 3:
            p0 = [1]
            bounds = ([-np.inf], [np.inf])
            func_name = 'LN'
        elif mode == 4:
            p0 = [0.45, 1, 1]
            bounds = ([0, -np.inf, -np.inf], [1, np.inf, np.inf])
            func_name = '2-LN'
        elif mode == 5:
            p0 = [0, 1]
            bounds = ([-np.inf, 0], [np.inf, np.inf])
            func_name = 'NN'
        elif mode == 6:
            p0 = [0.45, 0, 0, 1, 1]
            bounds = ([0, -np.inf, -np.inf, 0, 0], [1, np.inf, np.inf, np.inf, np.inf])
            func_name = '2-NN'
        elif mode == 7:
            p0 = [1]
            bounds = ([0], [np.inf])
            func_name = 'GA'
        elif mode == 8:
            p0 = [0.45, 1, 1]
            bounds = ([0, 0, 0], [1, np.inf, np.inf])
            func_name = '2-GA'
        elif mode == 9:
            p0 = [1]
            bounds = ([0], [np.inf])
            func_name = 'PO'
        elif mode == 10:
            p0 = [0.45, 1, 1]
            bounds = ([0, 0, 0], [1, np.inf, np.inf])
            func_name = '2-PO'
        elif mode == 11:
            p0 = [1, 2]
            bounds = ([0, 0], [np.inf, np.inf])
            func_name = 'AD'
        elif mode == 12:
            p0 = [1, 2]
            bounds = ([0, 0], [np.inf, np.inf])
            func_name = 'PD'
        elif mode == 13:
            p0 = [1, 1]
            bounds = ([0, 0], [np.inf, np.inf])
            func_name = 'ED'
        elif mode == 14:
            p0 = [0.5, 1, 1, 2, 0.5]
            bounds = ([0, 0, 0, 0, 0], [1, np.inf, np.inf, np.inf, np.inf])
            func_name = '2-ED'
        xaxis = np.asarray(xaxis)
        func = self.curve_fit_mapping(mode)
        try:
            fit_sigma = np.linspace(0.001, 0.01, len(xaxis)) # we want the points with larger value has larger weights,
                                                     # this is suitable for non-continuous metrics like AP and nDCG
            #print fit_sigma
            popt, pcov = curve_fit(func, xaxis, yaxis, p0=p0, method='trf', bounds=bounds, sigma=fit_sigma)
            perr = np.sqrt(np.diag(pcov))
            trialY = func(xaxis, *popt)
            #print mode, popt, np.absolute(trialY-yaxis).sum(), scipy.stats.ks_2samp(yaxis, trialY)
        except:
            return None
        return [mode, func_name, popt, trialY, np.absolute(trialY-yaxis).sum(), scipy.stats.ks_2samp(yaxis, trialY)]


class CalEstMAP(object):
    """
    compute the estimated MAP for the fitted models for relevant docs and non-relevant docs
    """

    def __init__(self):
        super(CalEstMAP, self).__init__()

    def cal_map(self, rel_docs=[], non_reldocs=[], all_docs=[], mode=1):
        """
        @mode: how to calculate the MAP
        1 - using discrete distributions for all docs and the rel docs. this is suitable for TF functions.
        2 - using continuous distributions for rel docs and non-rel docs.
        """
        if mode == 1:
            assert len(rel_docs) == len(all_docs)
            for i, ele in enumerate(rel_docs):
                if ele > all_docs[i]:
                    rel_docs[i] = all_docs[i]
            return EMAP().cal_expected_map(zip(rel_docs, all_docs))
        elif mode == 2:
            # TODO: implement
            return 0.0
        else:
            raise RuntimeError('mode must be in [1,2]')


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


class SD(object):
    """base class of 
    score distribution for rankings
    """
    __metaclass__ = ABCMeta
    def __init__(self, ranking_list, distribution_method, debug):
        """
        ranking_list for a specific query(one qid)
        """
        self._ranking_list, self._distribution_method, self._debug = \
            ranking_list, distribution_method, debug
        self._m1 = None
        self._v1 = None
        self._m0 = None
        self._v0 = None
        self._lambda = None
        self._aupr = None
        self._non_rel_distribution = None
        self._rel_distribution = None

    def _compute_stat_from_list(self, l):
        temp = np.array(l)
        mean = np.mean(temp)
        var = np.var(temp)
        return mean,var

    def _estimate_stats_with_rel_info(self):
        """estimate the statistics of relevant/non-relevant
        distributions(mean/variance: m/v). Note that the subscripts
        "1,0" corresponds statistics of relevant/non-relevant
        """
        nonrel_list = []
        rel_list = []
        for ele in self._ranking_list:
            score = ele[0]
            rel = ele[1]
            if rel:
                rel_list.append(score)
            else:
                nonrel_list.append(score)
        m1,v1 = self._compute_stat_from_list(rel_list)  
        m0,v0 = self._compute_stat_from_list(nonrel_list)
        estimated_lambda = len(rel_list)*1.0/len(self._ranking_list)

        self._m1 = m1
        self._v1 = v1
        self._m0 = m0
        self._v0 = v0
        self._lambda = estimated_lambda

        if self._debug :
            print "m1: %f, v1: %f, m0: %f, v0: %f" %(self._m1,self._v1,self._m0,self._v0)
            print "lambda: %f" %(estimated_lambda)
             
    def _compute_rel_likelihood(self, score):
        return self._rel_distribution.pdf(score)

    def _compute_nonrel_likelihood(self, score):
        return self._non_rel_distribution.pdf(score)

    def _compute_recall(self, score):
        return 1-self._rel_distribution.cdf(score)

    def _compute_fallout(self, score):
        return 1-self._non_rel_distribution.cdf(score)

    def _compute_aupr(self):
        lambda_value = self._lambda
        N = len(self._ranking_list)
        ap = .0
        s1 = self._ranking_list[0][0]
        if self._debug:
            print "top score is %f" % (s1)
        score = 2*s1
        recall = 0
        fallout = 0
        prec = [0]*N
        rec = [0]*N
        ds = score/N
        for i in range(N):
            score = score - ds
            #recall += self._compute_re_likelihood(qid,score)*ds
            #fallout += self._compute_non_re_likelihood(qid,score)*ds
            recall = self._compute_recall(score)
            fallout = self._compute_fallout(score)
            #if qid == "429":
            #    print "recall %f" %recall
            #    print "fallot %f" %fallout
            denominator = lambda_value*recall + (1-lambda_value)*fallout
            if recall == 0:
                prec[i] = 0
            else:
                prec[i] = (lambda_value*recall)/denominator
            rec[i] = recall
            if i>0:
                ap += (rec[i]-rec[i-1]) * (prec[i]+prec[i-1])/2
        if self._debug:
            print "ap = %f" % (ap)

        return ap 

    @property
    def aupr(self):
        if not self._aupr:
            raise RuntimeError("Parameters are not estimated!")
        else:
            
            return self._aupr

class GammaSD(SD):
    def __init__(self, run, debug=False):
        super(GammaSD, self).__init__(run,"gamma",debug)

    def _estimate_para(self, qrel=None):
        #estimate parameters for models
        self._estimate_stats_with_rel_info()
        self._k1 = (self._m1)**2 / self._v1
        self._theta1 = self._v1 / self._m1     
        self._k0 = (self._m0)**2 / self._v0
        self._theta0 = self._v0 / self._m0
        if self._debug :
            print "k1: %f, theta1: %f, k0: %f, theta0: %f" %(self._k1, self._theta1,self._k0,self._theta0)

    def estimate_distribution(self, qrel=None):
        self._estimate_para()
        self._rel_distribution = scipy.stats.gamma(self._k1,1/self._theta1) 
        self._non_rel_distribution = scipy.stats.gamma(self._k0,1/self._theta0)  

class LognormalSD(SD):
    def __init__(self,run,debug=False):
        super(LognormalSD,self).__init__(run,"lognormal",debug)

    def _estimate_para(self, qrel=None):
        #estimate parameters for models
        self._estimate_stats_with_rel_info()
        self._mu1 = math.log(self._m1+1e-6) - 0.5*(1 + (self._v1/(self._m1**2)) )
        var1 = math.log(1 + (self._v1/(self._m1**2)) )
        self._sigma1 = math.sqrt(var1)
        self._mu0 = math.log(self._m0+1e-6) - 0.5*(1 + (self._v0/(self._m0**2)) )
        var0 = math.log(1 + (self._v0/(self._m0**2)) )
        self._sigma0 = math.sqrt(var0)
        if self._debug :
            print "mu1: %f, sigma1: %f, mu0: %f, sigma0: %f" %(self._mu1,self._sigma1,self._mu0,self._sigma0)

    def estimate_distribution(self, qrel=None):
        self._estimate_para()
        self._rel_distribution = scipy.stats.lognorm(self._sigma1, scale = math.exp(self._mu1)) 
        self._non_rel_distribution = scipy.stats.lognorm(self._sigma0, scale = math.exp(self._mu0))


if __name__ == '__main__':
    #unittest.main()
    em = EM()
    a = [70,40,20,10,9,8,7,6,5,4,3,2,2,2,2,2,2,1,1,1,1,1]
    em.exponential(np.asarray(a)*1./np.sum(a))
