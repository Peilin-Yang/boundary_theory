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

class Metric(object):
    pass

class SVMMAP(Metric):
    def __init__(self, path):
        self.collection_path = os.path.abspath(path)
        if not os.path.exists(self.collection_path):
            frameinfo = getframeinfo(currentframe())
            print frameinfo.filename, frameinfo.lineno
            print '[TieBreaker Constructor]:Please provide a valid collection path'
            exit(1)

        self.svm_data_root = os.path.join(self.collection_path, 'svm_data')
        if not os.path.exists(self.svm_data_root):
            os.makedirs( self.svm_data_root )

        self.detailed_doc_stats_folder = os.path.join(self.collection_path, 'detailed_doc_stats')
        self.results_folder = os.path.join(self.collection_path, 'merged_results')
        self.eval_folder = os.path.join(self.collection_path, 'evals')

    def output_data_file(self):
        cs = CollectionStats(self.collection_path)
        single_queries = Query(self.collection_path).get_queries_of_length(1)
        queries = {ele['num']:ele['title'] for ele in single_queries}
        #print qids
        with open(os.path.join(self.collection_path, 'svm_data_index_file'), 'wb') as indexf:
            for qid in queries:
                data_fn = os.path.join(self.svm_data_root, qid)
                indexf.write('%s\n' % (data_fn))
                with open(data_fn, 'wb') as f:
                    for row in cs.get_qid_details(qid):
                        docid = row['docid']
                        total_tf = float(row['total_tf'])
                        doc_len = float(row['doc_len'])
                        rel_score = int(row['rel_score'])
                        #rel = (rel_score>=1)
                        f.write('%d qid:%s 1:%f 2:%f\n' % (rel_score, qid, total_tf, doc_len))
