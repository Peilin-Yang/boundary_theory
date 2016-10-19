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

    def cal_map(self, ranking_list, total_rel=0, type=1):
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
            if type == 0: # cal worst
                total += this_doc_cnt - rel_doc_cnt
            for j in range(rel_doc_cnt):
                cur_rel += 1 
                s += cur_rel*1.0/(total+j+1)
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
            title="", legend="", marker='ro'):
        # 1. probability distribution 
        ax.plot(xaxis, yaxis, marker, ms=4, label=legend)
        ax.vlines(xaxis, [0], yaxis)
        ax.set_title(title)
        ax.legend(loc='best')

    def plot_line(self, ax, xaxis, yaxis, 
            title="", legend="", legend_outside=False, marker=None, linestyle=None, 
            legend_pos='best', xlabel_format=0):
        ax.plot(xaxis, yaxis, marker=marker if marker else '+', 
            ls=linestyle if linestyle else '-', label=legend)
        ax.set_title(title)
        ax.legend(loc=legend_pos)
        if xlabel_format != 0:
            ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))

    def plot_bar(self, ax, xaxis, yaxis, title="", legend=""): 
        ax.bar(xaxis, yaxis, alpha=0.5, label=legend)
        ax.set_title(title)
        ax.legend(loc='best')

    def plot_hist(self, ax, yaxis, numbins=40, title="", legend=""): 
        ax.hist(yaxis, histtype='stepfilled', label=legend)
        ax.set_title(title)
        ax.legend(loc='best')

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

    def plot_interpolation(self, xaxis, yaxis, title, legend, output_fn, oformat='png'):
        """
        plot the interpolation figure
        """
        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(6, 3.*1))
        font = {'size' : 8}
        plt.rc('font', **font)
        #self.plot_dots(ax, xaxis, yaxis, title, legend) 
        self.plot_line(ax, xaxis, yaxis, title, legend) 
        #self.plot_bar(ax, xaxis, yaxis, title=title, legend=legend) 
        plt.savefig(output_fn, format=oformat, bbox_inches='tight', dpi=400)

    def interpolation_1(self, maxTF, type=1, oformat='png', tf_init=30, 
            tf_halflife=4, docs_cnt_init=2000, docs_cnt_halflife=1, 
            fit_larger_maxTF_cnt = 20, fit_larger_maxTF_type=1):
        """
        Normal distribution
        type:
            1 - set the number of documents as power decay
            2 - set the number of documents as radioactive decay
        """
        tf_init = float(tf_init)
        tf_halflife = float(tf_halflife)
        docs_cnt_init = float(docs_cnt_init)
        docs_cnt_halflife = float(docs_cnt_halflife)
        fit_larger_maxTF_cnt = int(fit_larger_maxTF_cnt)
        fit_larger_maxTF_type = int(fit_larger_maxTF_type)
        
        ranges = [i for i in range(1, maxTF+1)]
        if type == 1:
            tf_scale = [int(tf_init*math.pow(2, -1.*i/tf_halflife)) for i in ranges]
        elif type == 2:
            tf_scale = [tf_init*math.pow(i, -2) for i in ranges]
        docs_cnt_scale = [docs_cnt_init*math.pow(i, -2) for i in ranges]
        # elif type == 2:
        #     docs_cnt_scale = [int(docs_cnt_init*math.pow(2, -1.*i/docs_cnt_halflife)) for i in ranges]

        if fit_larger_maxTF_type != 0:
            for i in range(fit_larger_maxTF_cnt):
                ranges.append(maxTF+i+1)
                if fit_larger_maxTF_type == 1: # best
                    tf = 1 if i>=fit_larger_maxTF_cnt/2 else 0
                if fit_larger_maxTF_type == 2: # worst
                    tf = 1 if i<fit_larger_maxTF_cnt/2 else 0
                if fit_larger_maxTF_type == 3: # intermediate
                    tf = 1 if i%2==1 else 0
                tf_scale.append(tf)
                docs_cnt_scale.append(1)
        ranking_list = zip(tf_scale, docs_cnt_scale)
        print ranking_list
        performance = '1'
        # performance = 'best:' + str(round(self.cal_map(ranking_list, type=1), 4))
        # performance += '\nworst:' + str(round(self.cal_map(ranking_list, type=0), 4))
        title = r'TF-$R_0($%.1f)-$H$(%.1f)-$D_0$(%.1f)'  % (tf_init, tf_halflife, docs_cnt_init)
        verbose = 'interpolation1-%d-maxTF%d-tfinit%.1f-tfhalflife%.1f-docsinit%.1f-docshalflife%.1f-fitlargercnt%d-fitlargertype%d' \
            % (type, maxTF, tf_init, tf_halflife, docs_cnt_init, docs_cnt_halflife, fit_larger_maxTF_cnt, fit_larger_maxTF_type)
        output_fn = os.path.join(self.output_root, verbose+'.'+oformat) 
        yaxis = [ele[0]*1.0/ele[1] for ele in ranking_list]
        self.plot_interpolation(ranges, yaxis, title, performance, output_fn, oformat)

    def cal_map_with_interpolation(self, maxTF=20, interpolation_type=1, 
            subtype=1, oformat='png', interpolation_paras=[]):
        """
        Calculate the MAP with interpolation of the data.
        Typical interpolation is to change part of the y-axis. 
        For example, we can make the yaxis follows the Normal Distribution 
        where TF (xaxis) <= 20.
        """
        if interpolation_type == 1:
            self.interpolation_1(maxTF, subtype, oformat, *interpolation_paras)

