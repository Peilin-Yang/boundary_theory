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

    def cal_map(self, ranking_list, has_total_rel=False, total_rel=0):
        cur_rel = 0
        s = 0.0
        for i, ele in enumerate(ranking_list):
            docid = ele[0]
            rel = int(ele[1])>=1
            if rel:
                cur_rel += 1
                s += cur_rel*1.0/(i+1)
                if not has_total_rel:
                    total_rel += 1
        #print s/total_rel
        return s/total_rel

    def construct_relevance(self, type=1, maxTF=20):
        """
        Construct the relevance information.
        The return format is a list where the index of the list is the TF.
        Each element of the list is a tuple (numOfRelDocs, numOfTotalDocs) for 
        that TF.
        """
        if type == 1:
            return [(1, maxTF-i) for i in range(maxTF)]
        if type == 2:
            return [(1, maxTF-i) for i in range(maxTF+1)]

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
        # zoom
        if zoom:
            axins = inset_axes(ax,
                   width="50%",  # width = 30% of parent_bbox
                   height=0.8,  # height : 1 inch
                   loc=7) # center right
            zoom_xaxis = []
            for x in xaxis:
                if x <= 20:
                    zoom_xaxis.append(x)
            zoom_yaxis = yaxis[:len(zoom_xaxis)]
            axins.plot(zoom_xaxis, zoom_yaxis, marker, ms=4)
            axins.vlines(zoom_xaxis, [0], zoom_yaxis)
            axins.set_xlim(0, 20)
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")


    def plot_line(self, ax, xaxis, yaxis, 
            title="", legend="", legend_outside=False, marker=None, linestyle=None, 
            legend_pos='best',  xlabel_format=0):
        # 1. probability distribution 
        ax.plot(xaxis, yaxis, marker=marker if marker else '+', ls=linestyle if linestyle else '-', label=legend)
        ax.set_title(title)
        ax.legend(loc=legend_pos)
        if xlabel_format != 0:
            ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))

    def plot(self, plot_ratio=True, 
            performance_as_legend=True,  drawline=True, plotbins=True, numbins=60, 
            oformat='eps'):
        """
        plot the P(D=1|TF=x)

        Input:
        @x_func: how to get the x-axis of the figure. By default, this should 
            be TF values. But we are flexible with other options, e.g. tf/dl
        @_method: Which method is going to be plot. The parameters should also be 
            attached, e.g. dir,mu:2500
        @plot_ratio: When this is false, plot the y-axis as the number of relevant 
            documents; When this is true, plot the y-axis as the #rel_docs/#docs
        @performance_as_legend: whether to add performance(e.g. MAP) 
            as part of the legend
        @drawline: draw the data points as line(true) or dots(false)
        @plotbins: whether to group the x points as bins
        @numbins: the number of bins if we choose to plot x points as bins
        @oformat: output format, eps or png
        """
        fig, axs = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(6, 3.*1))
        font = {'size' : 8}
        plt.rc('font', **font)
        maxTF = 20
        xaxis = range(maxTF)
        yaxis = [ele[0]*1./ele[1] for ele in self.construct_relevance(1, maxTF)] 
        print yaxis
        if drawline:
            self.plot_line(axs, xaxis, yaxis)
        else:
            self.plot_dots(axs, xaxis, yaxis) 
        output_fn = os.path.join(self.output_root, 
            '%d-%s.%s' % (maxTF,
                'line' if drawline else 'dots', 
                oformat) )
        plt.savefig(output_fn, format=oformat, bbox_inches='tight', dpi=400)

PlotSyntheticMAP().plot()
