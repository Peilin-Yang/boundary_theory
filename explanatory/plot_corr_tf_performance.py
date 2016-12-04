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
from gen_doc_details import GenDocDetails
from rel_tf_stats import RelTFStats
from query import Query
from judgment import Judgment
from evaluation import Evaluation
from performance import Performances

import numpy as np
import scipy.stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.feature_selection import SelectFromModel, mutual_info_regression
from sklearn.model_selection import cross_val_score, cross_val_predict


class PlotCorrTFPeformance(object):
    """
    Plot the relationship between the tf in relevant docs with performance
    """
    def __init__(self, corpus_path, corpus_name):
        super(PlotCorrTFPeformance, self).__init__()

        self.collection_path = os.path.abspath(corpus_path)
        if not os.path.exists(self.collection_path):
            print '[Evaluation Constructor]:Please provide valid corpus path'
            exit(1)

        self.collection_name = corpus_name
        self.all_results_root = '../../all_results'
        if not os.path.exists(self.all_results_root):
            os.path.makedirs(self.all_results_root)
        self.rel_tf_stats_root = os.path.join(self.collection_path, 'rel_tf_stats')
        self.output_root = os.path.join(self.all_results_root, 'term_relationship')
        if not os.path.exists(self.output_root):
            os.makedirs(self.output_root)

    def get_data_with_label(self, all_data, label):
        req_data = [[all_data[qid]['terms'][t][label] for t in all_data[qid]['terms']] for qid in all_data]
        data = [[max(ele), min(ele), max(ele)-min(ele), (max(ele)+1e-8)/(min(ele)+1e-8), np.mean(ele), np.std(ele)] for ele in req_data]
        return np.array(data).transpose()

    def read_data(self, query_length=0):
        collection_name = self.collection_name
        cs = CollectionStats(self.collection_path)
        doc_details = GenDocDetails(self.collection_path)
        rel_tf_stats = RelTFStats(self.collection_path)
        if query_length == 0:
            queries = Query(self.collection_path).get_queries()
        else:
            queries = Query(self.collection_path).get_queries_of_length(query_length)
        queries = {ele['num']:ele['title'] for ele in queries}
        rel_docs = Judgment(self.collection_path).get_relevant_docs_of_some_queries(queries.keys(), 1, 'dict')
        queries = {k:v for k,v in queries.items() if k in rel_docs and len(rel_docs[k]) > 0}
        return RelTFStats(self.collection_path).get_data(queries.keys())

    def plot_all(self, query_length=0, oformat='png'):
        query_length = int(query_length)
        all_data = self.read_data(query_length)
        yaxis = [all_data[qid]['AP']['okapi'] for qid in all_data] # yaxis is the performance, e.g. AP
        all_xaxis = [
            ('tf_nonexisting_percent', self.get_data_with_label(all_data, 'zero_cnt_percentage')),
            ('tf_mean', self.get_data_with_label(all_data, 'mean')),
            ('tf_std', self.get_data_with_label(all_data, 'std')),
            ('df', self.get_data_with_label(all_data, 'df')),
            ('idf', self.get_data_with_label(all_data, 'idf')),
        ]

        # learning related start
        X, y = make_classification(n_samples=1000,
                           n_features=10,
                           n_informative=3,
                           n_redundant=0,
                           n_repeated=0,
                           n_classes=2,
                           random_state=0,
                           shuffle=False)
        #print X
        #print y
        all_data_samples = np.concatenate([np.transpose(ele[1]) for ele in all_xaxis], axis=1)
        forest = ExtraTreesRegressor(max_depth=2)
        forest.fit(all_data_samples, yaxis)
        print forest.score(all_data_samples, yaxis)
        #print forest.decision_path(all_data_samples)
        yfit = forest.predict(all_data_samples)
        #print zip(yaxis, yfit)
        importances = forest.feature_importances_
        print importances
        export_graphviz(forest, out_file=os.path.join(self.output_root, 'tree_%s-%d.dot' % (collection_name, query_length) ))
        lsvr = LinearSVR(C=0.01).fit(all_data_samples, yaxis)
        model = SelectFromModel(lsvr, prefit=True)
        X_new = model.transform(all_data_samples)
        print X_new.shape
        # learning related end



        xlabels = ['max', 'min', 'max-min', 'max/min', 'mean', 'std']
        num_cols = len(xlabels)
        num_rows = len(all_xaxis)
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, sharex=False, sharey=False, figsize=(3*num_cols, 3*num_rows))
        font = {'size' : 10}
        plt.rc('font', **font)
        row_idx = 0
        red_points = [[0,0], [0,0]]
        max_pearsonr = -1
        min_pearsonr = 1
        for i, ele in enumerate(all_xaxis):
            for j, xaxis in enumerate(ele[1]):
                pr = scipy.stats.pearsonr(xaxis, yaxis)[0]
                if pr > max_pearsonr:
                    max_pearsonr = pr
                    red_points[1] = [i, j]
                if pr < min_pearsonr:
                    min_pearsonr = pr
                    red_points[0] = [i, j]
        for i, ele in enumerate(all_xaxis):
            col_idx = 0
            for j, xaxis in enumerate(ele[1]):
                if num_rows > 1:
                    ax = axs[row_idx][col_idx]
                else:
                    if num_cols > 1:
                        ax = axs[col_idx]
                    else:
                        ax = axs
                zipped = zip(xaxis, yaxis)
                zipped.sort(key=itemgetter(0))
                xaxis_plot = zip(*zipped)[0]
                yaxis_plot = zip(*zipped)[1]
                legend = 'pearsonr:%.4f' % (scipy.stats.pearsonr(xaxis_plot, yaxis_plot)[0])
                if [i, j] in red_points:
                    if [i, j] == red_points[0]:
                        ax.plot(xaxis_plot, yaxis_plot, marker='o', markerfacecolor='r', ms=4, ls='None', label=legend)
                    else:
                        ax.plot(xaxis_plot, yaxis_plot, marker='o', markerfacecolor='g', ms=4, ls='None', label=legend)
                else:
                    ax.plot(xaxis_plot, yaxis_plot, marker='o', ms=4, ls='None', label=legend)
                ax.set_title(ele[0])
                if i == len(all_xaxis) - 1:
                    ax.set_xlabel(xlabels[j])
                ax.legend(loc='best', markerscale=0.5, fontsize=8)
                ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
                col_idx += 1
            row_idx += 1

        fig.suptitle(self.collection_name + ',qLen=%d' % query_length)
        output_fn = os.path.join(self.output_root, '%s-%d.%s' % (self.collection_name, query_length, oformat) )
        plt.savefig(output_fn, format=oformat, bbox_inches='tight', dpi=400)
