# -*- coding: utf-8 -*-
import sys,os
import math
import argparse
import json
import ast
import copy
import re
from operator import itemgetter
from subprocess import Popen, PIPE

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils/'))
from collection_stats import CollectionStats
from results_file import ResultsFile
from gen_doc_details import GenDocDetails
from rel_tf_stats import RelTFStats
from query import Query
from judgment import Judgment
from evaluation import Evaluation
from performance import Performances
from plot_corr_tf_performance import PlotCorrTFPeformance

import numpy as np
import scipy.stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class PlotTermRelationship(PlotCorrTFPeformance):
    """
    Plot the relationship between the tf in relevant docs with performance
    """
    def __init__(self, corpus_path, corpus_name):
        super(PlotTermRelationship, self).__init__(corpus_path, corpus_name)

    def plot_all(self, query_length=2, oformat='png'):
        query_length = int(query_length)
        all_data = self.read_data(query_length)
        zero_cnt_percentage = [[all_data[qid]['terms'][t]['zero_cnt_percentage'] for t in all_data[qid]['terms']] for qid in all_data]
        all_rel_cnts = [all_data[qid]['rel_cnt'] for qid in all_data]
        rel_contain_all_terms = [np.count_nonzero()==query_length for ele in zero_cnt_percentage]
        #rel_contain_one_term = [np.count_nonzero()==1 for ele in zero_cnt_percentage]
        #rel_contain_theother_term = [np.count_nonzero()==1 for ele in zero_cnt_percentage]
        print rel_contain_all_terms
