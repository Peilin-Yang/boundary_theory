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


class PlotTermRelationship(object):
    """
    Plot the relationship between the tf in relevant docs with performance
    """
    def __init__(self, corpus_path, corpus_name):
        super(PlotTermRelationship, self).__init__()
        self.collection_path = os.path.abspath(corpus_path)
        if not os.path.exists(self.collection_path):
            print '[Evaluation Constructor]:Please provide valid corpus path'
            exit(1)

        self.collection_name = corpus_name
        self.all_results_root = '../../all_results'
        if not os.path.exists(self.all_results_root):
            os.path.makedirs(self.all_results_root)
        self.rel_tf_stats_root = os.path.join(self.collection_path, 'rel_tf_stats')
        self.split_results_root = os.path.join(self.collection_path, 'split_results')
        self.output_root = os.path.join(self.all_results_root, 'term_relationship')
        if not os.path.exists(self.output_root):
            os.makedirs(self.output_root)


    def read_rel_data(self, query_length=0):
        rel_tf_stats = RelTFStats(self.collection_path)
        if query_length == 0:
            queries = Query(self.collection_path).get_queries()
        else:
            queries = Query(self.collection_path).get_queries_of_length(query_length)
        queries = {ele['num']:ele['title'] for ele in queries}
        rel_docs = Judgment(self.collection_path).get_relevant_docs_of_some_queries(queries.keys(), 1, 'dict')
        queries = {k:v for k,v in queries.items() if k in rel_docs and len(rel_docs[k]) > 0}
        return rel_tf_stats.get_data(queries.keys())

    def read_docdetails_data(self, query_length=2):
        if query_length == 0:
            queries = Query(self.collection_path).get_queries()
        else:
            queries = Query(self.collection_path).get_queries_of_length(query_length)
        queries = {ele['num']:ele['title'] for ele in queries}
        rel_docs = Judgment(self.collection_path).get_relevant_docs_of_some_queries(queries.keys(), 1, 'dict')
        queries = {k:v for k,v in queries.items() if k in rel_docs and len(rel_docs[k]) > 0}
        all_data = {}
        doc_details = GenDocDetails(self.collection_path)
        for qid in queries:
            all_data[qid] = doc_details.get_only_rels(qid)
        return all_data

    def rel_mapping(self, ele, dfs):
        """
        ele is the tfs of a doc, e.g. "this is a query" -> docid 155 -> [1, 2, 0, 3]
        """
        if np.count_nonzero(ele) == ele.size:
            return 3
        if np.count_nonzero(ele) == 1:
            if dfs[0] > dfs[1] and ele[1] > 0 or dfs[1] > dfs[0] and ele[0] > 0:
                return 2
            else:
                return 1
        return 0

    def prepare_rel_data(self, query_length, details_data, rel_data):
        """
        data is read from doc_details
        """
        countings = {}
        for qid in details_data:
            terms = details_data[qid][0]
            all_tfs = details_data[qid][1]
            dfs = details_data[qid][2]
            doc_lens = details_data[qid][3]
            tf_in_docs = all_tfs.transpose()
            mapped = []
            for ele in tf_in_docs:
                mapped.append(self.rel_mapping(ele, dfs))
            unique, counts = np.unique(mapped, return_counts=True)
            countings[qid] = {value: {'cnt':counts[i], 'rel_ratio': counts[i]*1./rel_data[qid]['rel_cnt']} for i, value in enumerate(unique)}
            for i in range(1, 4):
                if i not in countings[qid]:
                    countings[qid][i] = {'cnt': 0, 'rel_ratio': 0}
            if 0 not in countings[qid]:
                countings[qid][0] = rel_data[qid]['rel_cnt'] - sum([countings[qid][v]['cnt'] for v in countings[qid]])
        return countings


    def plot_all(self, query_length=2, oformat='png'):
        query_length = int(query_length)
        details_data = self.read_docdetails_data(query_length)
        rel_data = self.read_rel_data(query_length)
        prepared_data = self.prepare_rel_data(query_length, details_data, rel_data)
        all_xaxis = [[[prepared_data[qid][i][t] for t in prepared_data[qid][i]] for i in range(4)] for qid in prepared_data]
        print all_xaxis
        exit()
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
        yaxis = [float(all_data[qid]['AP']['okapi'][1]) for qid in all_data] # yaxis is the performance, e.g. AP
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
            markers = ['*', 's', '^', 'o']
            colors = ['k', 'r', 'g', 'b']
            legends = ['none', 'l-idf', 'h-idf', 'all']
            draw_legend = [False, False, False, False]
            print xaxis_plot, yaxis_plot
            xidx = 1
            for x,y in zip(xaxis_plot, yaxis_plot):
                if not draw_legend[x]:
                    draw_legend[x] = True
                    ax.plot(xidx, y, marker=markers[x], mfc=colors[x], ms=4, ls='None', label=legends[x])
                else:
                    ax.plot(xidx, y, marker=markers[x], mfc=colors[x], ms=4, ls='None')
                xidx += 1
            ax.set_title(label)
            ax.set_xlabel('queries')
            ax.set_xticklabels(qids_plot)
            ax.legend(loc='best', markerscale=0.5, fontsize=8)
            col_idx += 1

        fig.suptitle(self.collection_name + ',qLen=%d' % query_length)
        output_fn = os.path.join(self.output_root, '%s-%d.%s' % (self.collection_name, query_length, oformat) )
        plt.savefig(output_fn, format=oformat, bbox_inches='tight', dpi=400)

