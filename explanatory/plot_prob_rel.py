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

from emap import EMAP
from curve_fitting import EM
from curve_fitting import RealModels, FittingModels, CalEstMAP

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils/'))
from collection_stats import CollectionStats
from gen_doc_details import GenDocDetails
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


class PlotRelProb(object):
    """
    Plot the probability distribution of P(D=1|f(tf,dl,other_stats)=x)
    """
    def __init__(self, corpus_path, corpus_name):
        super(PlotRelProb, self).__init__()

        self.collection_path = os.path.abspath(corpus_path)
        if not os.path.exists(self.collection_path):
            print '[Evaluation Constructor]:Please provide valid corpus path'
            exit(1)

        self.collection_name = corpus_name
        self.all_results_root = '../../all_results'
        if not os.path.exists(self.all_results_root):
            os.path.makedirs(self.all_results_root)

    def plot_figure(self, ax, xaxis, yaxis, title='', legend='', 
            drawline=True, legend_outside=False, marker=None, 
            linestyle=None, xlabel='', ylabel='', xlog=False, ylog=False, 
            zoom=False, zoom_ax=None, zoom_xaxis=[], zoom_yaxis=[], 
            legend_pos='best', xlabel_format=0, xlimit=0, ylimit=0, legend_markscale=1.0):
        if drawline:
            ax.plot(xaxis, yaxis, marker=marker if marker else '+', ls=linestyle if linestyle else '-', label=legend)
        else: #draw dots 
            ax.plot(xaxis, yaxis, marker=marker if marker else 'o', markerfacecolor='r', ms=4, ls='None', label=legend)
            #ax.vlines(xaxis, [0], yaxis)
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
                zoom_ax.plot(zoom_xaxis, zoom_yaxis, marker=marker if marker else 'o', markerfacecolor='r', ms=4, ls='None')
                #zoom_ax.vlines(zoom_xaxis, [0], zoom_yaxis)
            if new_zoom_ax:
                zoom_ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
                mark_inset(ax, zoom_ax, loc1=2, loc2=4, fc="none", ec="0.5")
        return ax, zoom_ax

    def plot_rel_prob(self, query_length, x_func, _method, plot_ratio=True, 
            plot_total_or_avg=True, plot_rel_or_all=True, 
            performance_as_legend=True, drawline=True, numbins=60, xlimit=0, 
            ylimit=0, zoom_x=0, compact_x=False, curve_fitting=False, 
            draw_individual=False, draw_all=True, oformat='eps'):
        """
        plot the P(D=1|TF=x)

        Input:
        @query_length: only plot the queries of length, 0 for all queries.
        @x_func: how to get the x-axis of the figure. By default, this should 
            be TF values. But we are flexible with other options, e.g. tf/dl
        @_method: Which method is going to be plot. The parameters should also be 
            attached, e.g. dir,mu:2500
        @plot_ratio: When this is false, plot the y-axis as the number of relevant 
            documents; When this is true, plot the y-axis as the #rel_docs/#docs
        @plot_total_or_avg: When this is true, plot the y-axis as the collection 
            total ; When this is false, plot the collection average. 
            Only available when plot_ratio is false is only available for collection-wise
        @plot_rel_or_all: When this is true, plot the y-axis as the number of 
            relevant docs ; When this is false, plot the number of all docs. 
            Only available when plot_ratio is false is only available for collection-wise
        @performance_as_legend: whether to add performance(e.g. MAP) 
            as part of the legend
        @drawline: draw the data points as line(true) or dots(false)
        @numbins: the number of bins if we choose to plot x points as bins, 0 for no bins
        @xlimit: the limit of xaxis, any value larger than this value would not 
            be plotted. default 0, meaning plot all data.
        @ylimit: the limit of yaxis, any value larger than this value would not 
            be plotted. default 0, meaning plot all data.
        @zoom: whether zoom part of the plot
        @zoom_x: the zoom start x point, 0 for no zoom.
        @compact_x: map the x to continuous integers, e.g. 1,2,3,4,....
        @oformat: output format, eps or png
        """
        collection_name = self.collection_name
        cs = CollectionStats(self.collection_path)
        doc_details = GenDocDetails(self.collection_path)
        output_root = os.path.join('collection_figures', str(query_length))
        if not os.path.exists(os.path.join(self.all_results_root, output_root)):
            os.makedirs(os.path.join(self.all_results_root, output_root))
        if query_length == 0:
            queries = Query(self.collection_path).get_queries()
        else:
            queries = Query(self.collection_path).get_queries_of_length(query_length)
        queries = {ele['num']:ele['title'] for ele in queries}
        #print qids
        rel_docs = Judgment(self.collection_path).get_relevant_docs_of_some_queries(queries.keys(), 1, 'dict')
        #print rel_docs
        queries = {k:v for k,v in queries.items() if k in rel_docs and len(rel_docs[k]) > 0}
        #print np.mean([len(rel_docs[qid]) for qid in rel_docs])
        eval_class = Evaluation(self.collection_path)
        print _method
        p = eval_class.get_all_performance_of_some_queries(
            method=_method,
            qids=queries.keys(), 
            return_all_metrics=False, 
            metrics=['map']
        )
        collection_x_dict = {}
        collection_level_maxX = 0.0
        num_cols = min(4, len(queries))
        num_rows = int(math.ceil(len(rel_docs)*1.0/num_cols))
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, sharex=False, sharey=False, figsize=(2*num_cols, 2*num_rows))
        font = {'size' : 5}
        plt.rc('font', **font)
        row_idx = 0
        col_idx = 0
        #idfs = [(qid, math.log(cs.get_term_IDF1(queries[qid]))) for qid in rel_docs]
        #idfs.sort(key=itemgetter(1))
        all_expected_maps = []
        if curve_fitting:
            all_fitting_results = [{'sr': [], 'ap':[], 'ap_diff':[]} for i in range(FittingModels().size())]
            all_fitting_performances = {}
        for qid in sorted(queries):
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
            query_term = queries[qid]
            maxTF = cs.get_term_maxTF(query_term)
            #idf = math.log(cs.get_term_IDF1(query_term))
            #legend = 'idf:%.2f'%idf
            if performance_as_legend:
                legend = '\nAP:%.4f' % (p[qid]['map'] if p[qid] else 0)
            x_dict = {}
            qid_docs_len = 0
            #for row in cs.get_qid_details(qid):
            for row in doc_details.get_qid_details(qid):
                qid_docs_len += 1
                x = x_func(cs, row)
                if x > collection_level_maxX:
                    collection_level_maxX = x
                rel = (int(row['rel_score'])>=1)
                if x not in x_dict:
                    x_dict[x] = [0, 0] # [rel_docs, total_docs]
                if rel:
                    x_dict[x][0] += 1
                x_dict[x][1] += 1
                if x not in collection_x_dict:
                    collection_x_dict[x] = [0, 0] # [rel_docs, total_docs]
                if rel:
                    collection_x_dict[x][0] += 1
                collection_x_dict[x][1] += 1
            xaxis = x_dict.keys()
            xaxis.sort()
            xaxis = xaxis[:1000]
            if sum([x_dict[x][0] for x in xaxis]) == 0:
                continue
            if plot_ratio:
                yaxis = [x_dict[x][0]*1./x_dict[x][1] for x in xaxis]
            else:
                yaxis = [(x_dict[x][0]) if plot_rel_or_all else (x_dict[x][1]) for x in xaxis]
            ranking_list = [(x_dict[x][0], x_dict[x][1]) for x in xaxis]
            all_expected_maps.append(EMAP().cal_expected_map(ranking_list))

            if draw_individual:
                if np.sum(xaxis) == 0 or np.sum(yaxis) == 0:
                    continue
                raw_xaxis = copy.deepcopy(xaxis)
                xaxis = np.array(xaxis, dtype=np.float32)
                yaxis = np.array(yaxis, dtype=np.float32)
                if compact_x:
                    xaxis = range(1, len(xaxis)+1)
                if curve_fitting and not plot_ratio:
                    sum_yaxis = np.sum(yaxis)
                    yaxis /= sum_yaxis
                query_stat = cs.get_term_stats(query_term)
                zoom_xaxis = xaxis[zoom_x:]
                zoom_yaxis = yaxis[zoom_x:]
                ax, zoom_ax = self.plot_figure(ax, xaxis, yaxis, qid+'-'+query_term, legend,
                    drawline=drawline,
                    xlimit=xlimit,
                    ylimit=ylimit,
                    zoom=zoom_x > 0,
                    zoom_xaxis=zoom_xaxis,
                    zoom_yaxis=zoom_yaxis,
                    legend_markscale=0.5)
                if curve_fitting:
                    all_fittings = []
                    fitting_xaxis = []
                    fitting_yaxis = []
                    for i, ele in enumerate(yaxis):
                        if ele != 0:
                            fitting_xaxis.append(xaxis[i])
                            fitting_yaxis.append(ele)
                    for j in range(1, FittingModels().size()+1):
                        fitting = FittingModels().cal_curve_fit(fitting_xaxis, fitting_yaxis, j)
                        if not fitting is None:
                            fitting_func_name = fitting[1]
                            all_fitting_results[j-1]['name'] = fitting_func_name
                            all_fitting_results[j-1]['sr'].append(fitting[4]) # sum of squared error
                            if re.search(r'^tf\d+$', _method):
                                estimated_map = CalEstMAP().cal_map(
                                    #rel_docs = np.rint(fitting[3]*sum_yaxis).astype(int),
                                    rel_docs = np.rint(FittingModels().curve_fit_mapping(fitting[0])([x_dict[x][1] for x in raw_xaxis], *fitting[2])*sum_yaxis).astype(int),
                                    all_docs = [x_dict[x][1] for x in raw_xaxis],
                                    mode=1
                                )
                            else:
                                estimated_map = CalEstMAP().cal_map(
                                    rel_docs = np.rint(fitting[3]*sum_yaxis).astype(int),
                                    all_docs = [x_dict[x][1] for x in raw_xaxis],
                                    mode=1
                                )
                            all_fitting_results[j-1]['ap'].append(estimated_map) # average precision
                            actual_map = p[qid]['map'] if p[qid] else 0
                            all_fitting_results[j-1]['ap_diff'].append(math.fabs(estimated_map-actual_map))   
                            fitting.append(j) 
                            fitting.append(estimated_map)
                            fitting.append(math.fabs(estimated_map-actual_map))
                            all_fittings.append(fitting)
                            if fitting_func_name not in all_fitting_performances:
                                all_fitting_performances[fitting_func_name] = {}
                            all_fitting_performances[fitting_func_name][qid] = estimated_map
                            #print fitting[0], fitting[1], fitting[3]
                        else:
                            #print j, 'None'
                            pass
                    all_fittings.sort(key=itemgetter(4))
                    try:
                        print qid, query_term, all_fittings[0][0], all_fittings[0][1], all_fittings[0][2], all_fittings[0][4]
                    except:
                        continue
                    fit_curve_x = np.linspace(xaxis[0], xaxis[-1], 100)
                    fit_curve_y = FittingModels().curve_fit_mapping(all_fittings[0][-3])(fit_curve_x, *all_fittings[0][2])
                    # fitted_y = [0 for i in range(len(xaxis))]
                    # for x in xaxis:
                    #     if x in fitting_xaxis:
                    #         idx = fitting_xaxis.index(x)
                    #         fitted_y[idx] = all_fittings[0][3][idx]
                    best_fit_func_name = all_fittings[0][1]
                    all_fittings.sort(key=itemgetter(-1))
                    zoom_yaxis_fitting = fit_curve_y[zoom_x:]
                    self.plot_figure(ax, fit_curve_x, fit_curve_y, qid+'-'+query_term, 
                        '%s\n%s(%.4f)' % (best_fit_func_name, all_fittings[0][1], all_fittings[0][-2]), 
                        drawline=True, 
                        linestyle='--',
                        marker=" ", #nothing
                        zoom=zoom_x > 0,
                        zoom_ax = zoom_ax,
                        zoom_xaxis=zoom_xaxis,
                        zoom_yaxis=zoom_yaxis_fitting,
                        legend_pos='best',
                        xlimit=xlimit,
                        ylimit=ylimit,
                        legend_markscale=0.5)
        if draw_individual:
            output_fn = os.path.join(self.all_results_root, output_root, 
                '%s-%s-%s-%s-%s-%s-%d-%.1f-%.1f-zoom%d-%s-%s-individual.%s' % (
                    collection_name, 
                    _method, 
                    'ratio' if plot_ratio else 'abscnt', 
                    'total' if plot_total_or_avg else 'avg',
                    'rel' if plot_rel_or_all else 'all',
                    'line' if drawline else 'dots', 
                    numbins, 
                    xlimit,
                    ylimit,
                    zoom_x, 
                    'compact' if compact_x else 'raw',
                    'fit' if curve_fitting else 'plain',
                    oformat) )
            plt.savefig(output_fn, format=oformat, bbox_inches='tight', dpi=400)

            if curve_fitting:
                # plot the goodness of fit
                all_fitting_results = [ele for ele in all_fitting_results if 'name' in ele and ele['name'] not in ['AD']]
                goodness_fit_data = [ele['sr'] for ele in all_fitting_results if 'sr' in ele]
                labels = [ele['name'] for ele in all_fitting_results if 'sr' in ele and 'name' in ele]
                fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(6, 3.*1))
                font = {'size' : 8}
                plt.rc('font', **font)
                ax.boxplot(goodness_fit_data, labels=labels)
                output_fn = os.path.join(self.all_results_root, output_root, 
                    '%s-%s-fitting.%s' % (collection_name, _method, oformat) )
                plt.savefig(output_fn, format=oformat, bbox_inches='tight', dpi=400)
                # plot the AP diff
                ap_diff_data = [ele['ap_diff'] for ele in all_fitting_results if 'ap_diff' in ele]
                labels = [ele['name'] for ele in all_fitting_results if 'ap_diff' in ele and 'name' in ele]
                fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(6, 3.*1))
                font = {'size' : 8}
                plt.rc('font', **font)
                ax.boxplot(ap_diff_data, labels=labels)
                output_fn = os.path.join(self.all_results_root, output_root, 
                    '%s-%s-apdiff.%s' % (collection_name, _method, oformat) )
                plt.savefig(output_fn, format=oformat, bbox_inches='tight', dpi=400)

        # draw the figure for the whole collection
        collection_vocablulary_stat = cs.get_vocabulary_stats()
        collection_vocablulary_stat_str = ''
        idx = 1
        for k,v in collection_vocablulary_stat.items():
            collection_vocablulary_stat_str += k+'='+'%.2f'%v+' '
            if idx == 3:
                collection_vocablulary_stat_str += '\n'
                idx = 1
            idx += 1

        fig, axs = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(6, 3.*1))
        font = {'size' : 8}
        plt.rc('font', **font)
        xaxis = collection_x_dict.keys()
        xaxis.sort()
        if plot_ratio:
            yaxis = [collection_x_dict[x][0]*1./collection_x_dict[x][1] for x in xaxis]
        else:
            if plot_total_or_avg:
                yaxis = [(collection_x_dict[x][0]) if plot_rel_or_all else (collection_x_dict[x][1]) for x in xaxis] 
            else:
                yaxis = [(collection_x_dict[x][0]/len(queries)) if plot_rel_or_all else (collection_x_dict[x][1]/len(queries)) for x in xaxis]
            #print np.sum(yaxis[20:]), np.sum(yaxis[20:])
        if numbins > 0:
            interval = collection_level_maxX*1.0/numbins
            newxaxis = [i for i in np.arange(0, collection_level_maxX+1e-10, interval)]
            newyaxis = [[0.0, 0.0] for x in newxaxis]
            for x in xaxis:
                newx = int(x / interval)
                newyaxis[newx][0] += collection_x_dict[x][0]
                newyaxis[newx][1] += collection_x_dict[x][1]
            xaxis = newxaxis
            if plot_ratio:
                yaxis = [ele[0]/ele[1] if ele[1] != 0 else 0.0 for ele in newyaxis]
            else:
                if plot_total_or_avg:
                    yaxis = [(ele[0]) if plot_rel_or_all else (ele[1]) for ele in newyaxis] 
                else:
                    yaxis = [(ele[0]/len(queries)) if plot_rel_or_all else (ele[1]/len(queries)) for ele in newyaxis]

        # we do not care about the actual values of x
        # so we just map the actual values to integer values
        return_data = copy.deepcopy(collection_x_dict)

        if curve_fitting:
            #### calculate the stats
            for fitting_func_name in all_fitting_performances:
                actual_maps = [p[qid]['map'] if p[qid] else 0 for qid in queries]
                estimated_maps = [all_fitting_performances[fitting_func_name][qid] if qid in all_fitting_performances[fitting_func_name] else 0 for qid in queries]
                print fitting_func_name, 
                print scipy.stats.pearsonr(actual_maps, estimated_maps),
                print scipy.stats.kendalltau(actual_maps, estimated_maps)
                print '-'*30

        if draw_all:
            if compact_x:
                xaxis = range(1, len(xaxis)+1)

            xaxis = np.array(xaxis, dtype=np.float32)
            yaxis = np.array(yaxis, dtype=np.float32)
            if curve_fitting and not plot_ratio:
                yaxis /= np.sum(yaxis)

            collection_legend = ''
            if performance_as_legend:
                collection_legend = '$MAP:%.4f$' % (np.mean([p[qid]['map'] if p[qid] else 0 for qid in queries]))
                #collection_legend += '\n$MAP_E:%.4f$' % (np.mean(all_expected_maps))

            zoom_xaxis = xaxis[zoom_x:]
            zoom_yaxis = yaxis[zoom_x:]
            axs, zoom_axs = self.plot_figure(axs, xaxis, yaxis, collection_name, collection_legend, 
                drawline=drawline,
                xlimit=xlimit,
                ylimit=ylimit,
                zoom=zoom_x > 0,
                zoom_xaxis=zoom_xaxis,
                zoom_yaxis=zoom_yaxis)

            if curve_fitting:
                all_fittings = []
                fitting_xaxis = []
                fitting_yaxis = []
                for i, ele in enumerate(yaxis):
                    if ele != 0:
                        fitting_xaxis.append(xaxis[i])
                        fitting_yaxis.append(ele)
                for j in range(1, FittingModels().size()+1):
                    fitting = FittingModels().cal_curve_fit(fitting_xaxis, fitting_yaxis, j)
                    if not fitting is None:
                        all_fittings.append(fitting)
                        #print fitting[0], fitting[1], fitting[3]
                    else:
                        #print j, 'None'
                        pass
                all_fittings.sort(key=itemgetter(4))
                if all_fittings:
                    print all_fittings[0][0], all_fittings[0][1], all_fittings[0][2], all_fittings[0][4]
                    fitted_y = [0 for i in range(len(xaxis))]
                    for x in xaxis:
                        if x in fitting_xaxis:
                            idx = fitting_xaxis.index(x)
                            fitted_y[idx] = all_fittings[0][3][idx]

                    zoom_yaxis_fitting = fitted_y[zoom_x:]
                    self.plot_figure(axs, xaxis, fitted_y, collection_name, all_fittings[0][1], 
                        drawline=True, 
                        linestyle='--',
                        zoom=zoom_x > 0,
                        zoom_ax = zoom_axs,
                        zoom_xaxis=zoom_xaxis,
                        zoom_yaxis=zoom_yaxis_fitting,
                        legend_pos='best',
                        xlimit=xlimit,
                        ylimit=ylimit)

            output_fn = os.path.join(self.all_results_root, output_root, 
                '%s-%s-%s-%s-%s-%s-%d-%.1f-%.1f-zoom%d-%s-%s-all.%s' % (
                    collection_name, 
                    _method, 
                    'ratio' if plot_ratio else 'abscnt', 
                    'total' if plot_total_or_avg else 'avg',
                    'rel' if plot_rel_or_all else 'all',
                    'line' if drawline else 'dots', 
                    numbins, 
                    xlimit,
                    ylimit,
                    zoom_x, 
                    'compact' if compact_x else 'raw',
                    'fit' if curve_fitting else 'plain',
                    oformat) )
            plt.savefig(output_fn, format=oformat, bbox_inches='tight', dpi=400)

        return collection_name, return_data

    def wrapper(self, query_length, method_name, method_para_str, 
            plot_ratio, plot_total_or_avg, plot_rel_or_all, 
            performance_as_legend, drawline, numbins, xlimit, ylimit, 
            zoom_x, compact_x, fit_curve, draw_individual, draw_all, oformat='eps'):
        """
        This is the wrapper of the actual function. 
        We parse the CLI arguments and convert them to the values required 
        for the function.
        """
        x_func, formal_method_name = RealModels().get_func_mapping(method_name, method_para_str)
        return self.plot_rel_prob(
            int(query_length),
            x_func,
            formal_method_name,
            False if plot_ratio == '0' else True,
            False if plot_total_or_avg == '0' else True,
            False if plot_rel_or_all == '0' else True,
            False if performance_as_legend == '0' else True,
            False if drawline == '0' else True,
            int(numbins),
            float(xlimit),
            float(ylimit),
            int(zoom_x),
            False if compact_x == '0' else True,
            False if fit_curve == '0' else True,
            False if draw_individual == '0' else True,
            False if draw_all == '0' else True,
            oformat
        )

    def plot_with_data_single(self, xaxis, yaxis, yaxis_fitting, title, legend, 
            xlabel, ylabel, output_fn, query_length, method_name, 
            plot_ratio, plot_total_or_avg, plot_rel_or_all, 
            performance_as_legend, drawline, numbins, xlimit, ylimit, 
            zoom_x=20, oformat='eps'):
        fig, axs = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(6, 3.*1))
        font = {'size' : 12}
        plt.rc('font', **font)
        zoom_xaxis = xaxis[zoom_x:]
        zoom_yaxis = yaxis[zoom_x:]
        ax, zoom_ax = self.plot_figure(axs, xaxis, yaxis, title, legend, 
            drawline=drawline, 
            xlabel=xlabel,
            ylabel=ylabel,
            zoom=zoom_x > 0,
            zoom_xaxis=zoom_xaxis,
            zoom_yaxis=zoom_yaxis,
            legend_pos='best',
            xlimit=xlimit,
            ylimit=ylimit
        )

        if not yaxis_fitting is None:
            zoom_yaxis_fitting = yaxis_fitting[zoom_x:]
            self.plot_figure(
                axs, 
                xaxis, 
                yaxis_fitting, 
                title, 
                legend, 
                drawline=True, 
                linestyle='--',
                xlabel=xlabel,
                ylabel=ylabel,
                zoom=zoom_x > 0,
                zoom_ax = zoom_ax,
                zoom_xaxis=zoom_xaxis,
                zoom_yaxis=zoom_yaxis_fitting,
                legend_pos='best',
                xlimit=xlimit,
                ylimit=ylimit)

        plt.savefig(output_fn, format=oformat, bbox_inches='tight', dpi=400)

    def plot_with_data(self, data, title, legend, 
            query_length, method_name, method_para_str, plot_ratio, 
            plot_total_or_avg, plot_rel_or_all, performance_as_legend, 
            drawline, numbins, xlimit, ylimit, zoom_x=20, 
            compact_x=False, fit_curve=False, draw_individual=False, 
            draw_all=True, oformat='eps'):
        plot_ratio = False if plot_ratio == '0' else True
        plot_total_or_avg = False if plot_total_or_avg == '0' else True
        plot_rel_or_all = False if plot_rel_or_all == '0' else True
        performance_as_legend = False if performance_as_legend == '0' else True
        drawline = False if drawline == '0' else True
        numbins = int(numbins)
        xlimit = float(xlimit)
        ylimit = float(ylimit)
        zoom_x = int(zoom_x)
        compact_x = False if drawline == '0' else True
        compact_x = False if drawline == '0' else True
        x_func, formal_method_name = RealModels().get_func_mapping(method_name, method_para_str)
        

        xaxis = sorted(data.keys())
        yaxis = [[data[x][0]*1./data[x][1] for x in xaxis], [data[x][0] for x in xaxis], [data[x][1] for x in xaxis]]
        ylabels = ['ratio of rel docs', 'rel docs count', 'docs count']

        sum_rel = sum([data[x][0] for x in xaxis])
        sum_all = sum([data[x][1] for x in xaxis])
        y_prob = [[data[x][0]*1.0/sum_rel for x in xaxis], [data[x][1]*1.0/sum_all for x in xaxis]] 
        yprob_labels = ['PDF of rel docs', 'PDF of docs']
        
        if compact_x:
            xaxis = range(1, len(xaxis)+1)

        if fit_curve:
            y_prob_fitting = []
            for ele in y_prob:
                all_fittings = []
                fitting_xaxis = []
                fitting_yaxis = []
                for j, y in enumerate(ele):
                    if y != 0:
                        fitting_xaxis.append(xaxis[j])
                        fitting_yaxis.append(y)
                for k in range(1, FittingModels().size()+1):
                    fitting = FittingModels().cal_curve_fit(fitting_xaxis, fitting_yaxis, k)
                    if not fitting is None:
                        all_fittings.append(fitting)
                all_fittings.sort(key=itemgetter(4))
                try:
                    print all_fittings[0][0], all_fittings[0][1], all_fittings[0][2], all_fittings[0][4]
                except:
                    continue
                fitted_y = [0 for i in range(len(xaxis))]

                for x in xaxis:
                    if x in fitting_xaxis:
                        idx = fitting_xaxis.index(x)
                        fitted_y[idx] = all_fittings[0][3][idx]
                y_prob_fitting.append(fitted_y)

        output_root = os.path.join('collection_figures', query_length)
        if not os.path.exists(os.path.join(self.all_results_root, output_root)):
            os.makedirs(os.path.join(self.all_results_root, output_root))
        for i in range(3):
            output_fn = os.path.join(self.all_results_root, output_root, 
                '%s-%s-%s-%s-%s-%s-%d-%.1f-%.1f-zoom%d-%s-%s.%s' % (
                    'all_collections', 
                    formal_method_name, 
                    'ratio' if i==0 else 'abscnt', 
                    'total',
                    'rel' if i==1 else 'all',
                    'line' if drawline else 'dots', 
                    numbins, 
                    xlimit,
                    ylimit, 
                    zoom_x, 
                    'compact' if compact_x else 'raw',
                    'fit' if fit_curve else 'plain',
                    oformat) )
            self.plot_with_data_single(xaxis, yaxis[i], None, title, legend, 
                'projected doc score', ylabels[i], output_fn, query_length, 
                formal_method_name, plot_ratio, plot_total_or_avg, 
                plot_rel_or_all, performance_as_legend, drawline, numbins, 
                xlimit, ylimit, zoom_x, oformat)
        for i in range(2):
            output_fn = os.path.join(self.all_results_root, output_root, 
                '%s-%s-%s-%s-%s-%d-%.1f-%.1f-zoom%d-%s-%s.%s' % (
                    'all_collections', 
                    formal_method_name, 
                    'rel_dist' if i==0 else 'all_dist', 
                    'total',
                    'line' if drawline else 'dots', 
                    numbins, 
                    xlimit,
                    ylimit, 
                    zoom_x, 
                    'compact' if compact_x else 'raw',
                    'fit' if fit_curve else 'plain',
                    oformat) )
            self.plot_with_data_single(xaxis, y_prob[i], 
                y_prob_fitting[i] if fit_curve else None, 
                title, legend, 'projected doc score', yprob_labels[i], 
                output_fn, query_length, formal_method_name, 
                plot_ratio, plot_total_or_avg, plot_rel_or_all, 
                performance_as_legend, drawline, numbins, 
                xlimit, ylimit, zoom_x, oformat)
