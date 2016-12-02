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
from base import SingleQueryAnalysis
from collection_stats import CollectionStats
from query import Query
from judgment import Judgment
from evaluation import Evaluation
from performance import Performances
from gen_doc_details import GenDocDetails

import numpy as np
import scipy.stats


class Prints(object):
    """
    Prints all kinds of information
    """

    def __init__(self, path):
        super(Prints, self).__init__()
        self.collection_path = os.path.abspath(path)
        if not os.path.exists(self.collection_path):
            frameinfo = getframeinfo(currentframe())
            print frameinfo.filename, frameinfo.lineno
            print '[TieBreaker Constructor]:Please provide a valid collection path'
            exit(1)

        self.detailed_doc_stats_folder = os.path.join(self.collection_path, 'detailed_doc_stats')
        self.results_folder = os.path.join(self.collection_path, 'merged_results')
        self.eval_folder = os.path.join(self.collection_path, 'evals')

    def print_best_performances(self, methods=[]):
        single_queries = Query(self.collection_path).get_queries_of_length(1)
        queries = {ele['num']:ele['title'] for ele in single_queries}
        cs = CollectionStats(self.collection_path)
        performance = Performances(self.collection_path)
        res = performance.gen_optimal_performances_queries(methods, queries.keys())
        print res


    def print_statistics(self, methods):
        single_queries = Query(self.collection_path).get_queries_of_length(1)
        queries = {ele['num']:ele['title'] for ele in single_queries}
        cs = CollectionStats(self.collection_path)
        performance = Performances(self.collection_path)
        res = performance.gen_optimal_performances_queries(methods, queries.keys())

        avdl = cs.get_avdl()
        total_terms = cs.get_total_terms()
        collection_freq = []
        for qid in queries:
            idx = 0
            ctf = cs.get_term_collection_occur(queries[qid])
            idf = cs.get_term_logidf1(queries[qid])
            collection_freq.append( ctf*1.0/total_terms )
        print avdl
        print np.mean(collection_freq)

        for ele in res:
            label = ele[0]
            p = ele[1]
            para = float(ele[2].split(':')[1])
            print label
            if 'okapi' in label:
                print 'b:', para, 'beta:', 1.2*para/avdl, 'c2:', 1.2*(1-para)
            if 'pivoted' in label:
                print 's:', para, 'beta:', para/avdl, 'c2:', 1-para
    
    def batch_output_rel_tf_stats_paras(self):
        """
        Output the term frequency of query terms for relevant documents.
        For example, a query {journalist risks} will output 
        {
            'journalist': {'mean': 6.8, 'std': 1.0}, the average TF for journalist in relevant docs is 6.8
            'risks': {'mean': 1.6, 'std': 5.0}
        }
        """
        paras = []
        output_root = os.path.join(self.collection_path, 'rel_tf_stats')
        if not os.path.exists(output_root):
            os.makedirs(output_root)
        paras.append((self.collection_path))
        return paras

    def print_rel_tf_stats(self):
        queries = Query(self.collection_path).get_queries()
        queries = {ele['num']:ele['title'] for ele in queries}
        cs = CollectionStats(self.collection_path)
        doc_details = GenDocDetails(self.collection_path)
        for qid in queries:
            #if not os.path.exists(os.path.join(output_root, qid)):
            terms, tfs, dfs, doclens = doc_details.get_only_rels(qid)
            tf_mean = np.mean(tfs, axis=1)
            tf_std = np.std(tfs, axis=1)
            idfs = np.log((cs.get_doc_counts() + 1)/(dfs+1e-4))
            #try:
            okapi_perform = Performances(self.collection_path).gen_optimal_performances_queries('okapi', [qid])[0][1]
            terms_stats = {t:{'mean': tf_mean[idx], 'std': tf_std[idx], 
                'df': dfs[idx], 'idf': idfs[idx], 
                'zero_cnt_percentage': round(1.0-np.count_nonzero(tfs[idx])*1./tfs[idx].size, 2)} for idx, t in enumerate(terms) if dfs[idx] != 0}
            output = {
                'AP': {'okapi': okapi_perform},
                'terms': terms_stats
            }
            output_root = os.path.join(self.collection_path, 'rel_tf_stats')
            with open(os.path.join(output_root, qid), 'w') as f:
                json.dump(output, f, indent=2)
