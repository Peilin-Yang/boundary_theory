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
import datetime
import itertools

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils/'))
from base import SingleQueryAnalysis
from collection_stats import CollectionStats
from query import Query
from judgment import Judgment
from evaluation import Evaluation

import numpy as np
import scipy.stats

using_mpi = False

if using_mpi:
    from mpi4py import MPI # we have to use MPI in order to make the computation faster
    comm = MPI.COMM_WORLD

class Metric(object):
    pass

class MAP(Metric):
    def cal(self, ranking_list, has_total_rel=False, total_rel=-1, cut_off=1000):
        # if not has_total_rel:
        #     total_rel = 0
        # print datetime.datetime.now()
        # cur_rel = 0
        # s = 0.0
        # for i, ele in enumerate(ranking_list):
        #     # if i >= cut_off:
        #     #     break
        #     docid = ele[0]
        #     rel = ele[1]
        #     if rel:
        #         cur_rel += 1
        #         s += cur_rel*1.0/(i+1)
        #         if not has_total_rel:
        #             total_rel += 1
        # print s/total_rel
        # print datetime.datetime.now()
        # return s/total_rel
        a = np.array([int(ele[1]) for ele in ranking_list])
        non_zero = np.nonzero(a)[0]
        #print non_zero
        if not has_total_rel:
            total_rel = len(non_zero)
        cs = np.arange(1, len(non_zero)+1, 1.0)
        #print cs
        r = cs/(non_zero+1)
        #raw_input()
        return np.sum(r)/total_rel
    
    def get_nonzero_list(self, ranking_list, has_total_rel=False, total_rel=-1, cut_off=1000):
        a = np.array([int(ele[1]) for ele in ranking_list])
        non_zero = np.nonzero(a)[0]
        #print non_zero
        if not has_total_rel:
            total_rel = len(non_zero)
        cs = np.arange(1, len(non_zero)+1, 1.0)
        return non_zero

    def delta_map(self, idx1, idx2, nonzero):
        print indexes
        raw_input()


class RankingFunc(object):
    def __init__(self, kwargs):
        self.features = kwargs

    def cartesian(self, arrays, out=None):
        """
        Generate a cartesian product of input arrays.

        Parameters
        ----------
        arrays : list of array-like
            1-D arrays to form the cartesian product of.
        out : ndarray
            Array to place the cartesian product in.

        Returns
        -------
        out : ndarray
            2-D array of shape (M, len(arrays)) containing cartesian products
            formed of input arrays.

        Examples
        --------
        >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
        array([[1, 4, 6],
               [1, 4, 7],
               [1, 5, 6],
               [1, 5, 7],
               [2, 4, 6],
               [2, 4, 7],
               [2, 5, 6],
               [2, 5, 7],
               [3, 4, 6],
               [3, 4, 7],
               [3, 5, 6],
               [3, 5, 7]])

        """

        arrays = [np.asarray(x) for x in arrays]
        dtype = arrays[0].dtype

        n = np.prod([x.size for x in arrays])
        if out is None:
            out = np.zeros([n, len(arrays)], dtype=dtype)

        m = n / arrays[0].size
        out[:,0] = np.repeat(arrays[0], m)
        if arrays[1:]:
            self.cartesian(arrays[1:], out=out[0:m,1:])
            for j in xrange(1, arrays[0].size):
                out[j*m:(j+1)*m,1:] = out[0:m,1:]
        return out

    def delta_map_atom(self, indexes, nonzero_list):
        r_idx = indexes[0]
        nr_idx = indexes[1]
        min_idx = min(indexes)
        max_idx = max(indexes)
        if max_idx - min_idx == 1:
            masked = np.array([])
        else:
            masked = np.ma.masked_outside(nonzero_list, min_idx+1, max_idx-1).compressed()
        # print np.sum(1.0/masked), type(masked)
        np.seterr(divide='raise')
        try:
            delta = np.sum(1.0/(masked+1.0))
        except:
            print r_idx, nr_idx, nonzero_list, masked
            exit()
        # raw_input()
        # print r_idx, nr_idx
        # if r_idx == max_idx:
        #     print r_idx, nr_idx
        #     print nonzero_list, masked
        if r_idx == min_idx:
            # print r_idx, nr_idx
            #print nonzero_list, masked
            condition = np.where(nonzero_list>max_idx)
            try:
                idx = condition[0][0]
            except:
                idx = len(nonzero_list)
            #print idx
            diff = (np.where(nonzero_list==min_idx)[0][0]+1)*1.0/(min_idx+1)-idx*1.0/(max_idx+1)
        else:
            condition = np.where(nonzero_list<min_idx)
            #print condition, type(condition)
            try:
                idx = len(condition[0])+1# if condition[0].size() != 0 else 1
            except:
                idx = 1
            #print idx, np.where(nonzero_list==max_idx)[0][0]+1
            diff = idx*1.0/(min_idx+1)-(np.where(nonzero_list==max_idx)[0][0]+1)*1.0/(max_idx+1)
            # print '2'*20
            # print np.where(nonzero_list<min_idx)[0][-1]+1, min_idx, np.where(nonzero_list==max_idx)[0][0]+1, max_idx 
        # if r_idx == max_idx:
        #     print diff, delta, (np.fabs(diff)+delta) / len(nonzero_list)
        #     raw_input()
        return (np.fabs(diff)+delta) / len(nonzero_list)

    def cal_delta_map(self, rel_list, nonrel_list, ranking_list, rel_docs_cnt):
        ranking_mapping = {ele[0]:i for i, ele in enumerate(ranking_list)}
        map_class = MAP()
        nonzero_list = map_class.get_nonzero_list(ranking_list, True, rel_docs_cnt)
        #delta_map = np.array([[MAP().cal(self.swap_ranking(ranking_list, ranking_mapping, d1['docid'], d2['docid']), True, rel_docs_cnt) for d2 in nonrel_list] for d1 in rel_list])
        np_r = np.array([ranking_mapping[d1['docid']] for d1 in rel_list])
        np_nr = np.array([ranking_mapping[d2['docid']] for d2 in nonrel_list])
        permutation = self.cartesian([np_r, np_nr])
        #print permutation
        return np.apply_along_axis(self.delta_map_atom, 1, permutation, nonzero_list)

class Dirichlet(RankingFunc):
    def __init__(self, kwargs):
        super(Dirichlet, self).__init__(kwargs)
        self.static_mu = float(kwargs['mu']) if 'mu' in kwargs else 0.0
        self.static_eta = float(kwargs['eta']) if 'eta' in kwargs else 1e4
        print 'Dirichlet Init --- mu: ', self.static_mu, ' eta:', '%g' % self.static_eta
        self.mu = self.static_mu
        self.eta = self.static_eta
        self.pw_C = self.features['ctf']*1.0/self.features['total_terms']
        self.all_mus = np.array([])

    def score(self, paras):
        tf = paras['tf']
        ln = paras['ln']
        _tf = tf+self.mu*self.pw_C
        _ln = ln+self.mu
        #print tf, _tf, ln, _ln
        return _tf/_ln

    def swk(self, paras):
        tf = paras['tf']
        ln = paras['ln']
        return (self.pw_C*(ln+self.mu)-tf-self.mu*self.pw_C)/math.pow(self.mu+ln, 2)

    def reset_para(self):
        self.mu = self.static_mu

    def get_para(self):
        return 'mu:'+str(self.mu)

    def update_para(self, rel_list, nonrel_list, ranking_list, rel_docs_cnt, sigma=1):
        """
        rel_list && nonrel_list: [{'docid': docid, 'tf': total_tf, 'ln': doc_len, 'score': score}]
        """
        if using_mpi:
            # Broadcast A from rank 0 to everybody
            comm.Barrier()
            ranking_mapping = {ele[0]:i for i, ele in enumerate(ranking_list)}
            #comm.Barrier()
            for i in range(len(rel_list)):
                for j in range(len(nonrel_list)):
                    idx = i*len(nonrel_list)+j
                    if comm.rank == idx%comm.size+1:
                        print comm.rank, rel_list[i]['docid'], nonrel_list[j]['docid'],
                        print MAP().cal(self.swap_ranking(ranking_list, ranking_mapping, rel_list[i]['docid'], nonrel_list[j]['docid']), True, rel_docs_cnt)
            #delta_map = np.array([[MAP().cal(self.swap_ranking(ranking_list, ranking_mapping, d1['docid'], d2['docid']), True, rel_docs_cnt) for d2 in nonrel_list] for d1 in rel_list])
            exit()
        else:
            np_r = np.array([d1['score'] for d1 in rel_list])
            np_nr = np.array([-d2['score'] for d2 in nonrel_list])
            delta_map = self.cal_delta_map(rel_list, nonrel_list, ranking_list, rel_docs_cnt).reshape((len(rel_list), len(nonrel_list)))
            all_lambda = (sigma*(1.0)/(1+np.exp(sigma*(np.sum(self.cartesian([np_r, np_nr]), axis=1))))).reshape((len(rel_list), len(nonrel_list)))
            all_lambda *= delta_map
            # print np_r
            # print np_nr
            # print delta_map
            # print all_lambda
            # print len(np_r), len(np_nr), len(delta_map), len(all_lambda)
            lambda_rel = np.sum(np.sum(all_lambda, axis=1)*[self.swk(d) for d in rel_list])
            lambda_nonrel = np.sum(np.sum(all_lambda, axis=0)*[self.swk(d) for d in nonrel_list])
            delta_lambda = -lambda_rel + lambda_nonrel
            self.mu += self.eta * delta_lambda
            duplicates = np.ma.masked_values(self.all_mus, self.mu)
            if duplicates.size == 0:
                self.all_mus = np.append(self.all_mus, self.mu)
                self.eta = self.static_eta
            else:
                self.mu -= self.eta * delta_lambda
                self.eta /= 10.0
                self.mu += self.eta * delta_lambda
            print 'delta_lambda:', delta_lambda, self.get_para(), 
            #raw_input()

class TFLN1(RankingFunc):
    def __init__(self, kwargs):
        super(TFLN1, self).__init__(kwargs)
        self.static_c1 = float(kwargs['c1']) if 'c1' in kwargs else 0.0
        self.static_c2 = float(kwargs['c2']) if 'c2' in kwargs else 0.0
        self.static_eta = float(kwargs['eta']) if 'eta' in kwargs else 1e4
        print 'TFLN1 Init --- c1: ', self.static_c1, 'c2: ', self.static_c2, ' eta:', '%g' % self.static_eta
        self.c1 = self.static_c1
        self.c2 = self.static_c2
        self.eta = self.static_eta

    def score(self, paras):
        tf = paras['tf']
        ln = paras['ln']
        _tf = tf+self.c1
        _ln = ln+self.c2
        #print tf, _tf, ln, _ln
        return _tf/_ln

    def swk(self, paras, arg='c1'):
        tf = paras['tf']
        ln = paras['ln']
        if arg == 'c1':
            return 1.0/(ln+self.c2)
        elif arg == 'c2':
            return -1.0*(tf+self.c1)/math.pow(ln+self.c2, 2)

    def reset_para(self):
        self.c1 = self.static_c1
        self.c2 = self.static_c2

    def get_para(self):
        return 'c1:'+str(self.c1)+',c2:'+str(self.c2)

    def update_para(self, rel_list, nonrel_list, ranking_list, rel_docs_cnt, sigma=1):
        """
        rel_list && nonrel_list: [{'docid': docid, 'tf': total_tf, 'ln': doc_len, 'score': score}]
        """
        np_r = np.array([d1['score'] for d1 in rel_list])
        np_nr = np.array([-d2['score'] for d2 in nonrel_list])
        delta_map = self.cal_delta_map(rel_list, nonrel_list, ranking_list, rel_docs_cnt).reshape((len(rel_list), len(nonrel_list)))
        all_lambda = (sigma*(1.0)/(1+np.exp(sigma*(np.sum(self.cartesian([np_r, np_nr]), axis=1))))).reshape((len(rel_list), len(nonrel_list)))
        all_lambda *= delta_map
        lambda_rel_c1 = np.sum(np.sum(all_lambda, axis=1)*[self.swk(d, 'c1') for d in rel_list])
        lambda_nonrel_c1 = np.sum(np.sum(all_lambda, axis=0)*[self.swk(d, 'c1') for d in nonrel_list])
        delta_lambda_c1 = -lambda_rel_c1 + lambda_nonrel_c1
        lambda_rel_c2 = np.sum(np.sum(all_lambda, axis=1)*[self.swk(d, 'c2') for d in rel_list])
        lambda_nonrel_c2 = np.sum(np.sum(all_lambda, axis=0)*[self.swk(d, 'c2') for d in nonrel_list])
        delta_lambda_c2 = -lambda_rel_c2 + lambda_nonrel_c2
        self.c1 += self.eta * delta_lambda_c1
        self.c2 += self.eta * delta_lambda_c2
        print 'delta_lambda_c1:', delta_lambda_c1, 'delta_lambda_c2:', delta_lambda_c2, self.get_para(),

class TFLN2(RankingFunc):
    def __init__(self, kwargs):
        super(TFLN2, self).__init__(kwargs)
        self.static_c1 = float(kwargs['c1']) if 'c1' in kwargs else 0.0
        self.static_c2 = float(kwargs['c2']) if 'c2' in kwargs else 0.0
        self.static_alpha = float(kwargs['alpha']) if 'alpha' in kwargs else 0.0
        self.static_beta = float(kwargs['beta']) if 'beta' in kwargs else 0.0
        self.static_eta = float(kwargs['eta']) if 'eta' in kwargs else 1e4
        print 'TFLN2 Init --- alpha: ', self.static_alpha, 'beta: ', self.static_beta
        print 'TFLN2 Init --- c1: ', self.static_c1, 'c2: ', self.static_c2
        self.c1 = self.static_c1
        self.c2 = self.static_c2
        self.alpha = self.static_alpha
        self.beta = self.static_beta
        self.eta = self.static_eta

    def score(self, paras):
        tf = paras['tf']
        ln = paras['ln']
        _tf = self.alpha*tf+self.c1
        _ln = self.beta*ln+self.c2
        #print tf, _tf, ln, _ln
        return _tf/_ln

    def swk(self, paras, arg='c1'):
        tf = paras['tf']
        ln = paras['ln']
        if arg == 'alpha':
            return tf/(self.beta*ln+self.c2)
        elif arg == 'beta':
            return (self.alpha*tf+self.c1)*(-ln)/math.pow(self.beta*ln+self.c2, 2)
        elif arg == 'c1':
            return 1.0/(self.beta*ln+self.c2)
        elif arg == 'c2':
            return -1.0*(self.alpha*tf+self.c1)/math.pow(self.beta*ln+self.c2, 2)

    def reset_para(self):
        self.alpha = self.static_alpha
        self.beta = self.static_beta
        self.c1 = self.static_c1
        self.c2 = self.static_c2

    def get_para(self):
        return 'alpha:'+str(self.alpha)+',beta:'+str(self.beta)+',c1:'+str(self.c1)+',c2:'+str(self.c2)

    def update_para(self, rel_list, nonrel_list, ranking_list, rel_docs_cnt, sigma=1):
        """
        rel_list && nonrel_list: [{'docid': docid, 'tf': total_tf, 'ln': doc_len, 'score': score}]
        """
        np_r = np.array([d1['score'] for d1 in rel_list])
        np_nr = np.array([-d2['score'] for d2 in nonrel_list])
        delta_map = self.cal_delta_map(rel_list, nonrel_list, ranking_list, rel_docs_cnt).reshape((len(rel_list), len(nonrel_list)))
        all_lambda = (sigma*(1.0)/(1+np.exp(sigma*(np.sum(self.cartesian([np_r, np_nr]), axis=1))))).reshape((len(rel_list), len(nonrel_list)))
        all_lambda *= delta_map
        lambda_rel_alpha = np.sum(np.sum(all_lambda, axis=1)*[self.swk(d, 'alpha') for d in rel_list])
        lambda_nonrel_alpha = np.sum(np.sum(all_lambda, axis=0)*[self.swk(d, 'alpha') for d in nonrel_list])
        delta_lambda_alpha = -lambda_rel_alpha + lambda_nonrel_alpha
        lambda_rel_beta= np.sum(np.sum(all_lambda, axis=1)*[self.swk(d, 'beta') for d in rel_list])
        lambda_nonrel_beta = np.sum(np.sum(all_lambda, axis=0)*[self.swk(d, 'beta') for d in nonrel_list])
        delta_lambda_beta = -lambda_rel_beta + lambda_nonrel_beta
        lambda_rel_c1 = np.sum(np.sum(all_lambda, axis=1)*[self.swk(d, 'c1') for d in rel_list])
        lambda_nonrel_c1 = np.sum(np.sum(all_lambda, axis=0)*[self.swk(d, 'c1') for d in nonrel_list])
        delta_lambda_c1 = -lambda_rel_c1 + lambda_nonrel_c1
        lambda_rel_c2 = np.sum(np.sum(all_lambda, axis=1)*[self.swk(d, 'c2') for d in rel_list])
        lambda_nonrel_c2 = np.sum(np.sum(all_lambda, axis=0)*[self.swk(d, 'c2') for d in nonrel_list])
        delta_lambda_c2 = -lambda_rel_c2 + lambda_nonrel_c2
        self.alpha += self.eta * delta_lambda_alpha
        self.beta += self.eta * delta_lambda_beta
        self.c1 += self.eta * delta_lambda_c1
        self.c2 += self.eta * delta_lambda_c2
        print 'delta_lambda_alpha:', delta_lambda_alpha, 'delta_lambda_beta:', delta_lambda_beta, \
            'delta_lambda_c1:', delta_lambda_c1, 'delta_lambda_c2:', delta_lambda_c2, self.get_para(),

class TFLN3(RankingFunc):
    def __init__(self, kwargs):
        super(TFLN3, self).__init__(kwargs)
        self.static_c1 = float(kwargs['c1']) if 'c1' in kwargs else 0.0
        self.static_c2 = float(kwargs['c2']) if 'c2' in kwargs else 0.0
        self.static_alpha = float(kwargs['alpha']) if 'alpha' in kwargs else 1.0
        self.static_beta = float(kwargs['beta']) if 'beta' in kwargs else 1.0
        self.static_gamma = float(kwargs['gamma']) if 'gamma' in kwargs else 1.0
        self.static_eta = float(kwargs['eta']) if 'eta' in kwargs else 1e4
        print 'TFLN3 Init --- alpha: ', self.static_alpha, 'beta: ', self.static_beta
        print 'TFLN3 Init --- c1: ', self.static_c1, 'c2: ', self.static_c2
        self.c1 = self.static_c1
        self.c2 = self.static_c2
        self.alpha = self.static_alpha
        self.beta = self.static_beta
        self.gamma = self.static_gamma
        self.eta = self.static_eta

    def score(self, paras):
        tf = paras['tf']
        ln = paras['ln']
        _tf = self.alpha*tf+self.c1
        _ln = self.gamma*tf+self.beta*ln+self.c2
        #print tf, _tf, ln, _ln
        return _tf/_ln

    def swk(self, paras, arg='c1'):
        tf = paras['tf']
        ln = paras['ln']
        if arg == 'alpha':
            return tf/(self.gamma*tf+self.beta*ln+self.c2)
        elif arg == 'beta':
            return (self.alpha*tf+self.c1)*(-ln)/math.pow(self.gamma*tf+self.beta*ln+self.c2, 2)
        elif arg == 'gamma':
            return (self.alpha*tf+self.c1)*(-tf)/math.pow(self.gamma*tf+self.beta*ln+self.c2, 2)
        elif arg == 'c1':
            return 1.0/(self.gamma*tf+self.beta*ln+self.c2)
        elif arg == 'c2':
            return -1.0*(self.alpha*tf+self.c1)/math.pow(self.gamma*tf+self.beta*ln+self.c2, 2)

    def reset_para(self):
        self.alpha = self.static_alpha
        self.beta = self.static_beta
        self.gamma = self.static_gamma
        self.c1 = self.static_c1
        self.c2 = self.static_c2

    def get_para(self):
        return 'alpha:'+str(self.alpha)+',beta:'+str(self.beta)+',gamma:'+str(self.gamma)+',c1:'+str(self.c1)+',c2:'+str(self.c2)

    def update_para(self, rel_list, nonrel_list, ranking_list, rel_docs_cnt, sigma=1):
        """
        rel_list && nonrel_list: [{'docid': docid, 'tf': total_tf, 'ln': doc_len, 'score': score}]
        """
        np_r = np.array([d1['score'] for d1 in rel_list])
        np_nr = np.array([-d2['score'] for d2 in nonrel_list])
        delta_map = self.cal_delta_map(rel_list, nonrel_list, ranking_list, rel_docs_cnt).reshape((len(rel_list), len(nonrel_list)))
        all_lambda = (sigma*(1.0)/(1+np.exp(sigma*(np.sum(self.cartesian([np_r, np_nr]), axis=1))))).reshape((len(rel_list), len(nonrel_list)))
        all_lambda *= delta_map
        lambda_rel_alpha = np.sum(np.sum(all_lambda, axis=1)*[self.swk(d, 'alpha') for d in rel_list])
        lambda_nonrel_alpha = np.sum(np.sum(all_lambda, axis=0)*[self.swk(d, 'alpha') for d in nonrel_list])
        delta_lambda_alpha = -lambda_rel_alpha + lambda_nonrel_alpha
        lambda_rel_beta = np.sum(np.sum(all_lambda, axis=1)*[self.swk(d, 'beta') for d in rel_list])
        lambda_nonrel_beta = np.sum(np.sum(all_lambda, axis=0)*[self.swk(d, 'beta') for d in nonrel_list])
        delta_lambda_beta = -lambda_rel_beta + lambda_nonrel_beta
        lambda_rel_gamma = np.sum(np.sum(all_lambda, axis=1)*[self.swk(d, 'gamma') for d in rel_list])
        lambda_nonrel_gamma = np.sum(np.sum(all_lambda, axis=0)*[self.swk(d, 'gamma') for d in nonrel_list])
        delta_lambda_gamma = -lambda_rel_gamma + lambda_nonrel_gamma
        lambda_rel_c1 = np.sum(np.sum(all_lambda, axis=1)*[self.swk(d, 'c1') for d in rel_list])
        lambda_nonrel_c1 = np.sum(np.sum(all_lambda, axis=0)*[self.swk(d, 'c1') for d in nonrel_list])
        delta_lambda_c1 = -lambda_rel_c1 + lambda_nonrel_c1
        lambda_rel_c2 = np.sum(np.sum(all_lambda, axis=1)*[self.swk(d, 'c2') for d in rel_list])
        lambda_nonrel_c2 = np.sum(np.sum(all_lambda, axis=0)*[self.swk(d, 'c2') for d in nonrel_list])
        delta_lambda_c2 = -lambda_rel_c2 + lambda_nonrel_c2
        self.alpha += self.eta * delta_lambda_alpha
        self.beta += self.eta * delta_lambda_beta
        self.gamma += self.eta * delta_lambda_gamma
        self.c1 += self.eta * delta_lambda_c1
        self.c2 += self.eta * delta_lambda_c2
        print 'delta_lambda_alpha:', delta_lambda_alpha, 'delta_lambda_beta:', delta_lambda_beta, \
            'delta_lambda_c1:', delta_lambda_c1, 'delta_lambda_c2:', delta_lambda_c2, self.get_para(),

class LambdaRank(object):
    """
    Generate optimal parameters using LambdaRank
    """

    def __init__(self, path):
        self.collection_path = os.path.abspath(path)
        if not os.path.exists(self.collection_path):
            frameinfo = getframeinfo(currentframe())
            print frameinfo.filename, frameinfo.lineno
            print '[TieBreaker Constructor]:Please provide a valid collection path'
            exit(1)

        self.lambdarank_results_root = os.path.join(self.collection_path, 'lambdarank_results')
        if not os.path.exists(self.lambdarank_results_root):
            os.makedirs( self.lambdarank_results_root )

        self.detailed_doc_stats_folder = os.path.join(self.collection_path, 'detailed_doc_stats')
        self.results_folder = os.path.join(self.collection_path, 'merged_results')
        self.eval_folder = os.path.join(self.collection_path, 'evals')
        self.rel_docs = None

        self.sigma = 1
        self.max_iteration = 200
        self.stop_criteria = 1e-9

        self.last_performance = np.inf

    def learn(self, qid, data, collection_para, method_name, method_para_dict):
        """
        data = {True:[list of relevant docs], False:[list of non-relevant docs]}
        """
        method_factory = {
            'dir': Dirichlet,
            'tf_ln_1': TFLN1,
            'tf_ln_2': TFLN2,
        }
        collection_para.update(method_para_dict)
        m = method_factory[method_name](collection_para)
        self.last_performance = np.inf
        max_map = 0
        max_para = ''
        if (using_mpi and comm.rank == 0) or not using_mpi:  
            print 'collection:', self.collection_path
            print 'qid:', qid 
        for i in range(self.max_iteration):
            ranking = []
            for k,v in data.items():
                #print v
                for ele in v:
                    ele['score'] = m.score(ele)
                    ranking.append([ele['docid'], k, ele['score']])
            ranking.sort(key=itemgetter(2, 0), reverse=True)
            _map = MAP().cal(ranking, True, len(self.rel_docs[qid]))
            
            #raw_input()
            if _map > max_map:
                max_map = _map
                max_para = m.get_para()
            if np.isinf(self.last_performance):
                self.last_performance = _map
                if (using_mpi and comm.rank == 0) or not using_mpi:
                    print 'orig:', _map,
            else:
                #print 'last:', self.last_performance, 'this:', _map
                if (math.fabs(_map - self.last_performance) < self.stop_criteria):
                    break
                self.last_performance = _map
            print _map
            m.update_para(data[True], data[False], ranking, len(self.rel_docs[qid]), self.sigma)
            if using_mpi:
                comm.Barrier()
        print 
        print 'max:', max_map, 'para:', max_para
        return max_map, max_para


    def process(self, qid, method_name, method_paras, output_fn):
        cs = CollectionStats(self.collection_path)
        single_queries = Query(self.collection_path).get_queries_of_length(1)
        queries = {ele['num']:ele['title'] for ele in single_queries}
        #print qids
        self.rel_docs = Judgment(self.collection_path).get_relevant_docs_of_some_queries([qid], 1, 'dict')
        # idfs = [(qid, math.log(cs.get_term_IDF1(queries[qid]))) for qid in self.rel_docs]
        # idfs.sort(key=itemgetter(1))
        
        avdl = cs.get_avdl()
        total_terms = cs.get_total_terms()
        data = {True: [], False: []} # False: non-relevant  True: relevant

        ctf = cs.get_term_collection_occur(queries[qid])
        collection_para = {
            'avdl': avdl, 
            'total_terms': total_terms,
            'ctf': ctf
        }
        for row in cs.get_qid_details(qid):
            docid = row['docid']
            total_tf = float(row['total_tf'])
            doc_len = float(row['doc_len'])
            rel_score = int(row['rel_score'])
            rel = (rel_score>=1)
            data[rel].append( {
                'docid': docid,
                'tf': total_tf, 
                'ln': doc_len
            } )
        method_para_dict = {ele.split(':')[0]:ele.split(':')[1] for ele in method_paras.split(',')}
        max_map, max_para = self.learn(qid, data, collection_para, method_name, method_para_dict)
        with open(output_fn, 'wb') as f:
            json.dump({'map':max_map, 'para':max_para, 'eta':method_para_dict['eta']}, f, indent=2)

    def gen_lambdarank_paras(self, methods):
        paras = []
        single_queries = Query(self.collection_path).get_queries_of_length(1)
        queries = {ele['num']:ele['title'] for ele in single_queries}
        for qid in queries:
            for m in methods:
                for p in itertools.product(*[ele[1] for ele in m['paras'].items()]):
                    para_str = 'qid:%s-method:%s-' % (qid, m['name'])
                    method_para_list = []
                    for k_idx, k in enumerate(m['paras'].keys()):
                        #para_str += ',%s:%s' % (k, p[k_idx])
                        method_para_list.append('%s:%s' % (k, p[k_idx]))
                    method_para_str = ','.join(method_para_list)
                    results_fn = os.path.join(self.lambdarank_results_root, para_str+method_para_str)
                    if not os.path.exists(results_fn):
                        paras.append( (self.collection_path, \
                            qid, m['name'], method_para_str, results_fn) )
        return paras

    def print_results(self, print_details=False):
        res = {}
        for fn in os.listdir(self.lambdarank_results_root):
            fn_parts = fn.split('-')
            qid = fn_parts[0].split(':')[1]
            method = fn_parts[1].split(':')[1]
            with open(os.path.join(self.lambdarank_results_root, fn)) as f:
                r = json.load(f)
                eta = int(float(r['eta']))
                if method not in res:
                    res[method] = {}
                if eta not in res[method]:
                    res[method][eta] = {}
                if qid not in res[method][eta]:
                    res[method][eta][qid] = []
                res[method][eta][qid].append( (r['map'], r['para']) )
        for method in res:
            for eta in res[method]:
                for qid in res[method][eta]:
                    res[method][eta][qid].sort(key=itemgetter(0), reverse=True)

        for method in res:
            for eta in res[method]:
                all_res = []
                print '=' * 30
                print 'eta:%g' % eta
                for qid in res[method][eta]:
                    if print_details:
                        print method, qid, res[method][eta][qid][0]
                    all_res.append(res[method][eta][qid][0][0])
                print method, 'all', np.mean(all_res)

    def print_results_para(self, _method='dir'):
        res = {}
        for fn in os.listdir(self.lambdarank_results_root):
            fn_parts = fn.split('-')
            qid = fn_parts[0].split(':')[1]
            method = fn_parts[1].split(':')[1]
            if method != _method:
                continue
            with open(os.path.join(self.lambdarank_results_root, fn)) as f:
                r = json.load(f)
                eta = int(float(r['eta']))
                if eta not in res:
                    res[eta] = {}
                if qid not in res[eta]:
                    res[eta][qid] = []
                res[eta][qid].append( (r['map'], r['para']) )
        for eta in res:
            for qid in res[eta]:
                res[eta][qid].sort(key=itemgetter(0), reverse=True)

        eta_performances = []
        for eta in res:
            all_res = []
            for qid in res[eta]:
                all_res.append(res[eta][qid][0][0])
            eta_performances.append( (eta, np.mean(all_res) ) )
        eta_performances.sort(key=itemgetter(1), reverse=True)
        #print eta_performances
        best_eta = eta_performances[0][0]

        all_paras = {}
        print '=' * 30
        print 'eta:%g' % best_eta
        for qid in res[best_eta]:
            p = res[best_eta][qid][0][1]
            for para in p.split(','):
                para_key = para.split(':')[0]
                para_value = float(para.split(':')[1])
                if para_key not in all_paras:
                    all_paras[para_key] = []
                all_paras[para_key].append(para_value)
            if _method == 'tf_ln_1':
                if 'ratio' not in all_paras:
                    all_paras['ratio'] = []
                all_paras['ratio'].append(all_paras['c1'][-1]/all_paras['c2'][-1])
        for para_key in all_paras:
            print para_key, np.mean(all_paras[para_key]), np.std(all_paras[para_key])

