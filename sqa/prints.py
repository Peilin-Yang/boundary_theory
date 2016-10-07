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

            

    def get_docs_tf(self):
        """
        We get the statistics from /collection_path/detailed_doc_stats/ 
        so that we can get everything for the top 10,000 documents for 
        each query generated by Dirichlet language model method.
        """
        single_queries = Query(self.collection_path).get_queries_of_length(1)
        queries = {ele['num']:ele['title'] for ele in single_queries}
        cs = CollectionStats(self.collection_path)
        res = {}
        for qid in queries:
            res[qid] = []
            idx = 0
            for row in cs.get_qid_details(qid):
                docid = row['docid']
                tf = float(row['total_tf'])
                #score = self.hypothesis_tf_function(tf, _type, scale, mu, sigma)
                res[qid].append([docid, tf])
                idx += 1
                if idx >= 1000:
                    break
        return res

    def cut_docs_tf_with_maxTF(self, maxTF=20):
        docs_tf = self.get_docs_tf()
        for qid in docs_tf:
            tf = [ele for ele in docs_tf[qid] if ele[1] <= maxTF]
            tf.sort(key=itemgetter(1,0), reverse=True)
            docs_tf[qid] = tf
        return docs_tf

    def cal_map(self, ranking_list_with_judgement):
        cur_rel = 0
        s = 0.0
        total = 0
        for i, ele in enumerate(ranking_list_with_judgement):
            docid = ele[0]
            is_rel = ele[1]
            if is_rel:
                cur_rel += 1
            s += cur_rel*1.0/(i+1)
        if cur_rel == 0:
            return 0
        return s/cur_rel

    def print_map_with_cut_maxTF(self, maxTF=20):
        single_queries = Query(self.collection_path).get_queries_of_length(1)
        rel_docs = Judgment(self.collection_path).get_relevant_docs_of_some_queries([ele['num'] for ele in single_queries])
        cutted_docs = self.cut_docs_tf_with_maxTF(maxTF)
        for qid in cutted_docs:
            ranking_with_judge = [(doc[0], doc[0] in [ele[0] for ele in rel_docs[qid]]) for doc in cutted_docs[qid]]
            print qid, self.cal_map(ranking_with_judge)
