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
        self.output_root = os.path.join(self.all_results_root, 'term_relationship')
        if not os.path.exists(self.output_root):
            os.makedirs(self.output_root)

    def plot_all(self, query_length=2, oformat='png'):
        query_length = int(query_length)
        all_data = self.read_data(query_length)
        zero_cnt_percentage = [[all_data[qid]['terms'][t]['zero_cnt_percentage'] for t in all_data[qid]['terms']] for qid in all_data]
        highest_idf_term_idx = [np.argmax([all_data[qid]['terms'][t]['idf'] for t in all_data[qid]['terms']]) for qid in all_data]
        all_rel_cnts = [all_data[qid]['rel_cnt'] for qid in all_data]
        all_xaxis = []
        plot_data = []
        for i, ele in enumerate(zero_cnt_percentage):
            if np.count_nonzero(ele)==query_length:
                plot_data.append(3)
            elif (ele[0] != 0 and highest_idf_term_idx[i] == 0) \
                or (ele[1] != 0 and highest_idf_term_idx[i] == 1): 
                plot_data.append(2)
            elif ele[0] != 0 or ele[1] != 0:
                plot_data.append(1)
            else:
                plot_data.append(0)
        print plot_data
        all_xaxis.append(('dist_in_rel_docs', plot_data))
        yaxis = [all_data[qid]['AP']['okapi'] for qid in all_data] # yaxis is the performance, e.g. AP
        num_cols = 1
        num_rows = 1
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, sharex=False, sharey=False, figsize=(3*num_cols, 3*num_rows))
        font = {'size' : 10}
        plt.rc('font', **font)
        row_idx = 0
        for i, ele in enumerate(all_xaxis):
            col_idx = 0
            label = ele[0]
            xaxis = ele[1]
            if num_rows > 1:
                ax = axs[row_idx][col_idx]
            else:
                if num_cols > 1:
                    ax = axs[col_idx]
                else:
                    ax = axs
            zipped = zip(all_data.keys(), xaxis, yaxis)
            zipped.sort(key=itemgetter(2))
            qids_plot = np.array(zip(*zipped)[0])
            xaxis_plot = np.array(zip(*zipped)[1])
            yaxis_plot = np.array(zip(*zipped)[2])
            markers = ['o', 's', '*', '^']
            colors = ['r', 'g', 'b', 'k']
            legends = ['all', 'h-idf', 'l-idf', 'none']
            for x,y in zip(xaxis_plot, yaxis_plot):
                if x == 3:
                    ax.plot(x, y, marker=markers[x], mfc=colors[x], ms=4, ls='None', label=legends[x])
            ax.set_title(label)
            ax.set_xlabel('queries')
            ax.legend(loc='best', markerscale=0.5, fontsize=8)
            col_idx += 1

        fig.suptitle(self.collection_name + ',qLen=%d' % query_length)
        output_fn = os.path.join(self.output_root, '%s-%d.%s' % (self.collection_name, query_length, oformat) )
        plt.savefig(output_fn, format=oformat, bbox_inches='tight', dpi=400)

