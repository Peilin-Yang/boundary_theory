# -*- coding: utf-8 -*-
import sys,os
import math
import argparse
import json
import ast
from operator import itemgetter

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils/'))
from base import SingleQueryAnalysis
from collection_stats import CollectionStats
from query import Query
from judgment import Judgment
from evaluation import Evaluation
from performance import Performances

import numpy as np
from scipy.stats import norm, expon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class PlotSyntheticMAP(SingleQueryAnalysis):
    """
    This class plots the performance(MAP) for the synthetic collection.
    The purpose here is to see how the curve correlates with the performance
    """
    def __init__(self):
        super(PlotSyntheticMAP, self).__init__()

        self.output_root = os.path.join(self.all_results_root, 'synthetic_documents') 
        if not os.path.exists(self.output_root):
            os.makedirs(self.output_root)

    def cal_map(self, ranking_list, has_total_rel=False, total_rel=0, type=1):
        """
        Calculate the MAP based on the ranking_list.

        Input:
        @ranking_list: The format of the ranking_list is:
            [(num_rel_docs, num_total_docs), (num_rel_docs, num_total_docs), ...]
            where the index corresponds to the TF, e.g. ranking_list[1] is TF=1
        @type: 1 for best performance (relevant documents are ranked before 
            non-relevant documents for each TF) and 0 for worst performance (
            relevant documents are ranked after non-relevant documents for each TF)
        """
        cur_rel = 0
        s = 0.0
        total = 0
        for ele in reversed(ranking_list):
            rel_doc_cnt = ele[0]
            this_doc_cnt = ele[1]
            if rel_doc_cnt == 0:
                continue
            if type == 0: # cal worst
                total += this_doc_cnt - rel_doc_cnt
            for j in range(rel_doc_cnt):
                cur_rel += 1 
	        s += cur_rel*1.0/(total+j+1)
	        if not has_total_rel:
		    total_rel += 1
            total += this_doc_cnt if type == 1 else rel_doc_cnt
        #print s/total_rel
        if total_rel == 0:
            return 0
        return s/total_rel

    def construct_relevance(self, type=1, maxTF=20, scale_factor=1):
        """
        Construct the relevance information.
        The return format is a list where the index of the list is the TF.
        Each element of the list is a tuple (numOfRelDocs, numOfTotalDocs) for 
        that TF.
        """
        ranges = [i for i in range(1, maxTF+1)]
        if type == 1:
            l = [(1, (maxTF-i+1)) for i in ranges]
        if type == 2:
            l = [(3, (maxTF-i+3)) for i in ranges]
        if type == 3:
            l = [(int(round(i*10.0/maxTF, 0)), 10) for i in ranges]
        if type == 4:
            l = [(i-3 if i-3 >= 0 else 0, i) for i in ranges]
        if type == 5:
            l = [(i-1, i) for i in ranges]
        return [(ele[0]*scale_factor, ele[1]*scale_factor) for ele in l]

    def construct_relevance_impact(self, ranking_list, affect_idx=0, 
            rel_docs_change=10):
        """
        Construct the modified relevance information based on ranking list 
        that was generated by self.construct_relevance()
        """
        # make a list so that we can modify it
        l = [[ele[0], ele[1]] for ele in ranking_list]
        l[affect_idx][0] += rel_docs_change
        l[affect_idx][1] += rel_docs_change
        return l

    def plot_dots(self, ax, xaxis, yaxis, 
            title="", legend="", legend_outside=False, marker='ro', 
            xlog=True, ylog=False, zoom=False, legend_pos='upper right', 
            xlabel_format=0):
        # 1. probability distribution 
        ax.plot(xaxis, yaxis, marker, ms=4, label=legend)
        ax.vlines(xaxis, [0], yaxis)
        if xlog:
            ax.set_xscale('log')
        if ylog:
            ax.set_yscale('log')
        ax.set_xlim(0, ax.get_xlim()[1] if ax.get_xlim()[1]<100 else 100)
        #ax.set_ylim(0, ax.get_ylim()[1] if ax.get_ylim()[1]<500 else 500)
        ax.set_title(title)
        ax.legend(loc=legend_pos)
        if xlabel_format != 0:
            ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))

    def plot_line(self, ax, xaxis, yaxis, 
            title="", legend="", legend_outside=False, marker=None, linestyle=None, 
            legend_pos='best', xlabel_format=0):
        ax.plot(xaxis, yaxis, marker=marker if marker else '+', 
            ls=linestyle if linestyle else '-', label=legend)
        ax.set_title(title)
        ax.legend(loc=legend_pos)
        if xlabel_format != 0:
            ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))

    def plot(self, maxTF=20, scale_factor=1, oformat='png', plot_ratio=True, 
            performance_as_legend=True, drawline=True):
        """
        plot the relationship between P(D=1|TF=x) and the performance 
        on a synthetic data set

        Input:
        @plot_ratio: When this is false, plot the y-axis as the number of relevant 
            documents; When this is true, plot the y-axis as the #rel_docs/#docs
        @performance_as_legend: whether to add performance(e.g. MAP) 
            as part of the legend
        @drawline: draw the data points as line(true) or dots(false)
        @oformat: output format, eps or png
        """
        fig, axs = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(6, 3.*1))
        font = {'size' : 8}
        plt.rc('font', **font)
        markers = ['', 's', 's', '.', '+', '+']
        for plot_type in range(1, 6):
            xaxis = range(1, maxTF+1)
            ranking = self.construct_relevance(plot_type, maxTF, scale_factor)
            yaxis = [ele[0]*1./ele[1] for ele in ranking] 
            best_map = self.cal_map(ranking, type=1)
            worst_map = self.cal_map(ranking, type=0)
            legend = 'map: %.4f~%.4f' % (worst_map, best_map)
            if drawline:
                self.plot_line(axs, xaxis, yaxis, marker=markers[plot_type], legend=legend)
            else:
                self.plot_dots(axs, xaxis, yaxis, legend=legend) 
        output_fn = os.path.join(self.output_root, 
            '%d-%d-%d-%s.%s' % (plot_type, scale_factor, maxTF,
                'line' if drawline else 'dots', 
                oformat) )
        plt.savefig(output_fn, format=oformat, bbox_inches='tight', dpi=400)

    def plot_num_rel_docs_impact(self, maxTF=20, rel_docs_change=1, oformat='png'):
        """
        output the impact of changing the number of relevant documents at each 
        data point. The approach is to increase/decrease the number of relevant 
        documents for a specific data point and see what is the consequence of 
        doing so. 
        """
        fig, axs = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(6, 3.*1))
        font = {'size' : 8}
        plt.rc('font', **font)
        markers = ['', 's', 's', '.', '+', '+']
        for plot_type in range(1, 6):
            xaxis = range(1, maxTF+1)
            ranking = self.construct_relevance(plot_type, maxTF, 1)
            yaxis = []
            for i in range(len(ranking)):
                modified_ranking = self.construct_relevance_impact(ranking, i, rel_docs_change)
                best_map = self.cal_map(modified_ranking, type=1)
                worst_map = self.cal_map(modified_ranking, type=0)
                yaxis.append(worst_map)
            self.plot_line(axs, xaxis, yaxis, marker=markers[plot_type]) 
        output_fn = os.path.join(self.output_root, 
            'impact-%d-%d-%d.%s' % (plot_type, maxTF, rel_docs_change, oformat) )
        plt.savefig(output_fn, format=oformat, bbox_inches='tight', dpi=400)

    def interpolation_1(self, maxTF, type='1'):
        """
        Normal distribution
        type:
            1 - set the number of relevant documents as a contant scale scale_factor
            2 - set the number of relevant documents as exponential decay
        """
        if type == '2':
            tf_scale = [50/i for i in range(1, maxTF+1)]
            docs_cnt_scale = [2000/(i*i) for i in range(1, maxTF+1)]
        # if type == 2:
        #     l = [(3, (maxTF-i+3)) for i in ranges]
        # if type == 3:
        #     l = [(int(round(i*10.0/maxTF, 0)), 10) for i in ranges]
        # if type == 4:
        #     l = [(i-3 if i-3 >= 0 else 0, i) for i in ranges]
        # if type == 5:
        #     l = [(i-1, i) for i in ranges]
        # return [(ele[0]*scale_factor, ele[1]*scale_factor) for ele in l]
        print tf_scale
        print docs_cnt_scale

    def cal_map_with_interpolation(self, maxTF=20, interpolation_type=1, 
            interpolation_paras=[]):
        """
        Calculate the MAP with interpolation of the data.
        Typical interpolation is to change part of the y-axis. 
        For example, we can make the yaxis follows the Normal Distribution 
        where TF (xaxis) <= 20.
        """
        if interpolation_type == 1:
            self.interpolation_1(maxTF, *interpolation_paras)

