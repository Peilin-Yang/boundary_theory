# -*- coding: utf-8 -*-
import sys,os
import json
import csv
from operator import itemgetter
import numpy as np
import itertools
from subprocess import Popen, PIPE
from inspect import currentframe, getframeinfo
import argparse

reload(sys)
sys.setdefaultencoding('utf-8')


class Performances(object):
    """
    Handle the performace. For example, get all the performances of one method(has multiple parameters).
    When constructing, pass the path of the corpus. For example, "../wt2g/"
    """
    def __init__(self, collection_path):
        self.corpus_path = os.path.abspath(collection_path)
        if not os.path.exists(self.corpus_path):
            frameinfo = getframeinfo(currentframe())
            print frameinfo.filename, frameinfo.lineno
            print '[Evaluation Constructor]:Please provide a valid corpus path'
            exit(1)

        self.evaluation_results_root = os.path.join(self.corpus_path, 'evals')
        self.mb_decay_eval_results_root = os.path.join(self.corpus_path, 'evals_mb_decay')
        self.mb_combine_eval_results_root = os.path.join(self.corpus_path, 'evals_mb_combine')
        self.performances_root = os.path.join(self.corpus_path, 'performances')
        if not os.path.exists(self.performances_root):
            os.makedirs(self.performances_root)

    def gen_optimal_performances_queries(self, methods=[], qids=[], _query_part='title', eval_method='map'):
        all_results = {}
        for fn in os.listdir(self.evaluation_results_root):
            query_part = fn.split('-')[0]
            if query_part != _query_part:
                continue
            method_paras = '-'.join(fn.split('-')[1:])
            method_paras_split = method_paras.split(',')
            method_name = method_paras_split[0].split(':')[1]
            if method_name not in methods:
                continue
            if 'perturb' in method_name:
                method_paras_split = {ele.split(':')[0]:ele.split(':')[1] for ele in method_paras_split}
                label = query_part+'-'+method_name+'_'+method_paras_split['perturb_type']
            else:
                label = query_part+'-'+method_name
            compare_results_fn = os.path.join(self.performances_root, label)
            if label not in all_results:
                all_results[label] = []
            all_results[label].append( os.path.join(self.evaluation_results_root, fn) )
        #print all_results
        res = []
        for label in all_results:
            r = self.get_best_performances(all_results[label], qids)
            res.append( (label, round(r[0], 4), r[1]) )
        return res

    def get_best_performances(self, eval_fn_list, qids=[], eval_method='map'):
        if not qids:
            qids = ['all']
        all_results = {}
        for ele in eval_fn_list:
            paras = ','.join( '-'.join(os.path.basename(ele).split('-')[1:]).split(',')[1:] )
            if paras not in all_results:
                all_results[paras] = []
            with open(ele) as _in:
                j = json.load(_in)
                for qid in qids:
                    all_results[paras].append( (float(j[qid][eval_method]), qid) )
        
        final_results = []
        for paras in all_results:
            final_results.append((np.mean([ele[0] for ele in all_results[paras]]), paras))
        final_results.sort(key=itemgetter(0), reverse=True)
        return final_results[0]

    def load_optimal_performance(self, methods=[], evaluation_method='map', query_part='title'):
        data = []
        for fn in os.listdir(self.performances_root):
            q_part = fn.split('-')[0]
            if q_part != query_part:
                continue
            method_name = fn.split('-')[1]
            if (not methods) or (methods and method_name in methods):
                with open(os.path.join(self.performances_root, fn)) as pf:
                    all_performance = json.load(pf)
                    required = all_performance[evaluation_method]
                    data.append( (method_name, required['max']['value'], required['max']['para']) )
        return data

    def print_optimal_performance(self, methods=[], evaluation_method='map', query_part='title'):
        optimal_performances = self.load_optimal_performance(methods, evaluation_method, query_part)
        optimal_performances.sort(key=itemgetter(0, 1, 2))
        for ele in optimal_performances:
            print ele[0], ele[1], ele[2]


if __name__ == '__main__':
    pass

