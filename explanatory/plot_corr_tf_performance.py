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

    def plot_figure(self, ax, xaxis, yaxis, title='', legend='', 
            drawline=True, legend_outside=False, marker=None, 
            linestyle=None, xlabel='', ylabel='', xlog=False, ylog=False, 
            zoom=False, zoom_ax=None, zoom_xaxis=[], zoom_yaxis=[], 
            legend_pos='best', xlabel_format=0, xlimit=0, ylimit=0, legend_markscale=1.0):
        if drawline:
            ax.plot(xaxis, yaxis, marker=marker if marker else '+', ls=linestyle if linestyle else '-', label=legend)
        else: #draw dots 
            #ax.plot(xaxis, yaxis, marker=marker if marker else 'o', ms=4, ls='None', label=legend)
            ax.vlines(xaxis, [0], yaxis, label=legend)
        if xlog:
            ax.set_xscale('log')
        if ylog:
            ax.set_yscale('log')
        if xlimit > 0:
            ax.set_xlim(0, ax.get_xlim()[1] if ax.get_xlim()[1]<xlimit else xlimit)
        if ylimit > 0:
            ax.set_ylim(0, ax.get_ylim()[1] if ax.get_ylim()[1]<ylimit else ylimit)
        ax.set_title(title)
        ax.legend(loc=legend_pos, markerscale=legend_markscale)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        # zoom
        if zoom:
            new_zoom_ax = False
            if zoom_ax is None:
                new_zoom_ax = True  
                zoom_ax = inset_axes(ax,
                       width="70%",  # width = 50% of parent_bbox
                       height="70%",  # height : 1 inch
                       loc=7) # center right
            if drawline:
                zoom_ax.plot(zoom_xaxis, zoom_yaxis, marker=marker if marker else '+', ls=linestyle if linestyle else '-')
            else: #draw dots 
                #zoom_ax.plot(zoom_xaxis, zoom_yaxis, marker=marker if marker else 'o', markerfacecolor='r', ms=4, ls='None')
                zoom_ax.vlines(zoom_xaxis, [0], zoom_yaxis)
            if new_zoom_ax:
                zoom_ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
                mark_inset(ax, zoom_ax, loc1=2, loc2=4, fc="none", ec="0.5")
        return ax, zoom_ax

    def relation_least_appear_term_performance(self, all_data, query_length=0, oformat='png'):
        all_zero_cnts = [[all_data[qid]['terms'][t]['zero_cnt_percentage'] for t in all_data[qid]['terms']] for qid in all_data]
        data = [max(ele) for ele in all_zero_cnts]
        return data
    def relation_appear_max_min_diff_performance(self, all_data, query_length=0, oformat='png'):
        all_zero_cnts = [[all_data[qid]['terms'][t]['zero_cnt_percentage'] for t in all_data[qid]['terms']] for qid in all_data]
        data = [max(ele)-min(ele) for ele in all_zero_cnts]
        return data
    def relation_appear_max_min_ratio_performance(self, all_data, query_length=0, oformat='png'):
        all_zero_cnts = [[all_data[qid]['terms'][t]['zero_cnt_percentage'] for t in all_data[qid]['terms']] for qid in all_data]
        data = [(max(ele)+1e-8)/(min(ele)+1e-8) for ele in all_zero_cnts]
        return data
    def relation_appear_diff_mean_performance(self, all_data, query_length=0, oformat='png'):
        all_zero_cnts = [[all_data[qid]['terms'][t]['zero_cnt_percentage'] for t in all_data[qid]['terms']] for qid in all_data]
        data = [np.mean(ele) for ele in all_zero_cnts]
        return data
    def relation_appear_diff_std_performance(self, all_data, query_length=0, oformat='png'):
        all_zero_cnts = [[all_data[qid]['terms'][t]['zero_cnt_percentage'] for t in all_data[qid]['terms']] for qid in all_data]
        data = [np.std(ele) for ele in all_zero_cnts]
        return data

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
            ('least_appear_term_zero_ratio', self.relation_least_appear_term_performance(all_data, query_length, oformat)),
            ('appear_zero_diff', self.relation_appear_max_min_diff_performance(all_data, query_length, oformat)),
            ('appear_zero_ratio', self.relation_appear_max_min_ratio_performance(all_data, query_length, oformat)),
            ('appear_zero_mean', self.relation_appear_diff_mean_performance(all_data, query_length, oformat)),
            ('appear_zero_std', self.relation_appear_diff_std_performance(all_data, query_length, oformat)),
        ]

        num_cols = min(5, len(all_xaxis))
        num_rows = int(math.ceil(len(all_xaxis)*1.0/num_cols))
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, sharex=False, sharey=False, figsize=(2*num_cols, 2*num_rows))
        font = {'size' : 8}
        plt.rc('font', **font)
        row_idx = 0
        col_idx = 0
        for ele in all_xaxis:
            if num_rows > 1:
                ax = axs[row_idx][col_idx]
            else:
                if num_cols > 1:
                    ax = axs[col_idx]
                else:
                    ax = axs
            col_idx += 1
            if col_idx >= num_cols:
                row_idx += 1
                col_idx = 0
            xaxis = ele[1]
            zipped = zip(xaxis, yaxis)
            zipped.sort(key=itemgetter(0))
            xaxis_plot = zip(*zipped)[0]
            yaxis_plot = zip(*zipped)[1]
            legend = 'pearsonr:%.4f' % (scipy.stats.pearsonr(xaxis_plot, yaxis_plot)[0])
            ax.plot(xaxis_plot, yaxis_plot, marker='o', ms=4, ls='None', label=legend)
            ax.set_title(ele[0])

        fit.suptitle(self.collection_name + 'qLen=%d' % query_length)
        output_fn = os.path.join(self.output_root, '%s-%d.%s' % (self.collection_name, query_length, oformat) )
        plt.savefig(output_fn, format=oformat, bbox_inches='tight', dpi=400)
