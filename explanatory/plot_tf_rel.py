# -*- coding: utf-8 -*-
import sys,os
import math
import argparse
import json
import ast
import copy
from operator import itemgetter
from subprocess import Popen, PIPE

from em import EM

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils/'))
from collection_stats import CollectionStats
from gen_doc_details import GenDocDetails
from query import Query
from judgment import Judgment
from evaluation import Evaluation
from performance import Performances

import numpy as np
from sklearn.neighbors import KernelDensity
import scipy.stats
from scipy.stats import entropy
from scipy.optimize import curve_fit
from scipy.interpolate import spline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


class PlotTFRel(object):
    """
    Plot the probability distribution of P(D=1|f(tf,dl,other_stats)=x)
    """
    def __init__(self, corpus_path, corpus_name):
        super(PlotTFRel, self).__init__()

        self.collection_path = os.path.abspath(corpus_path)
        if not os.path.exists(self.collection_path):
            print '[Evaluation Constructor]:Please provide valid corpus path'
            exit(1)

        self.collection_name = corpus_name
        self.all_results_root = '../../all_results'
        if not os.path.exists(self.all_results_root):
            os.path.makedirs(self.all_results_root)

    def A(self, pr, pn, r, n):
        """
        """
        if r == 0:
           return 0
        R = {}
        for i in range(1, r+1):
            for j in range(n+1):
                R[(pr+r-i, pn+n-j, i, 0)] = (pr+r-i+1.0) / (pr+r-i+pn+n-j+1.0) 
                if i != 1:
                    R[(pr+r-i, pn+n-j, i, 0)] += R[(pr+r-i+1, pn+n-j, i-1, 0)]
        for i in range(1, r+1):
            for j in range(1, n+1):
                subR = R[(pr+r-i+1, pn+n-j, i-1, j)] if i!=1 else 0
                prob_r = i*1.0/(i+j)*((pr+r-i+1.0)/(pr+r-i+pn+n-j+1)+subR) 
                prob_n = j*1.0/(i+j)*R[(pr+r-i, pn+n-j+1, i, j-1)]
                R[(pr+r-i, pn+n-j, i, j)] = prob_r + prob_n
        return R[(pr, pn, r, n)]


    def cal_expected_map(self, ranking_list, total_rel=0):
        """
        Calculate the MAP based on the ranking_list.

        Input:
        @ranking_list: The format of the ranking_list is:
            [(num_rel_docs, num_total_docs), (num_rel_docs, num_total_docs), ...]
            where the index corresponds to the TF, e.g. ranking_list[1] is TF=1
        """
        s = 0.0
        pr = 0
        pn = 0
        for ele in reversed(ranking_list):
            rel_doc_cnt = ele[0]
            this_doc_cnt = ele[1]
            nonrel_doc_cnt = this_doc_cnt - rel_doc_cnt
            s += self.A(pr, pn, rel_doc_cnt, nonrel_doc_cnt)
            pr += rel_doc_cnt
            pn += nonrel_doc_cnt
            total_rel += rel_doc_cnt
        #print s/total_rel
        if total_rel == 0:
            return 0
        return s/total_rel

    def plot_single_tfc_constraints_draw_pdf_dot(self, ax, xaxis, yaxis, 
            title, legend, legend_outside=False, marker='ro', 
            xlabel='', ylabel='', xlog=False, ylog=False, zoom=False, 
            zoom_xaxis=[], zoom_yaxis=[], legend_pos='upper right', 
            xlabel_format=0, xlimit=0, ylimit=0):
        # 1. probability distribution 
        ax.plot(xaxis, yaxis, marker, ms=4, label=legend)
        ax.vlines(xaxis, [0], yaxis)
        if xlog:
            ax.set_xscale('log')
        if ylog:
            ax.set_yscale('log')
        if xlimit > 0:
            ax.set_xlim(0, ax.get_xlim()[1] if ax.get_xlim()[1]<xlimit else xlimit)
        if ylimit > 0:
            ax.set_ylim(0, ax.get_ylim()[1] if ax.get_ylim()[1]<ylimit else ylimit)
        ax.set_title(title)
        ax.legend(loc=legend_pos)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        # zoom
        if zoom:
            axins = inset_axes(ax,
                   width="70%",  # width = 50% of parent_bbox
                   height="70%",  # height : 1 inch
                   loc=7) # center right
            axins.plot(zoom_xaxis, zoom_yaxis, marker, ms=4)
            axins.vlines(zoom_xaxis, [0], zoom_yaxis)
            axins.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            #axins.set_xlim(0, 20)
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")


    def plot_single_tfc_constraints_draw_pdf_line(self, ax, xaxis, yaxis, 
            title, legend, legend_outside=False, marker=None, 
            linestyle=None, xlabel='', ylabel='', xlog=False, ylog=False, 
            zoom=False, zoom_xaxis=[], zoom_yaxis=[], legend_pos='upper right', 
            xlabel_format=0, xlimit=0, ylimit=0):
        # 1. probability distribution 
        ax.plot(xaxis, yaxis, marker=marker if marker else '+', ls=linestyle if linestyle else '-', label=legend)
        if xlog:
            ax.set_xscale('log')
        if ylog:
            ax.set_yscale('log')
        if xlimit > 0:
            ax.set_xlim(0, ax.get_xlim()[1] if ax.get_xlim()[1]<xlimit else xlimit)
        if ylimit > 0:
            ax.set_ylim(0, ax.get_ylim()[1] if ax.get_ylim()[1]<ylimit else ylimit)
        ax.set_title(title)
        ax.legend(loc=legend_pos)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        # zoom
        if zoom:
            axins = inset_axes(ax,
                   width="70%",  # width = 50% of parent_bbox
                   height="70%",  # height : 1 inch
                   loc=7) # center right
            axins.plot(zoom_xaxis, zoom_yaxis, marker, ms=4)
            axins.vlines(zoom_xaxis, [0], zoom_yaxis)
            axins.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            #axins.set_xlim(0, 20)
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")


    def hypothesis_tf_function(self, tf, mu, sigma, scale):
        """
        The Hypothesis for the TF function of disk45 collection.
        The function is :
        S = N (μ, σ )(μ = 5, σ = 2), if tf ≤ 10
        S = 1, if 10 < tf ≤ 20
        S = tf, if tf > 20
        """
        return scale+scipy.stats.norm(mu, sigma).pdf(tf)


    def get_docs_tf(self):
        """
        We get the statistics from /collection_path/detailed_doc_stats/ 
        so that we can get everything for the top 10,000 documents for 
        each query generated by Dirichlet language model method.
        """
        single_queries = Query(self.collection_path).get_queries_of_length(1)
        queries = {ele['num']:ele['title'] for ele in single_queries}
        cs = CollectionStats(self.collection_path)
        res = {}
        for qid in queries:
            res[qid] = []
            idx = 0
            for row in cs.get_qid_details(qid):
                docid = row['docid']
                tf = float(row['total_tf'])
                #score = self.hypothesis_tf_function(tf, _type, scale, mu, sigma)
                res[qid].append([docid, tf])
                idx += 1
                if idx >= 1000:
                    break
        return res
    
    def get_docs_len(self):
        """
        We get the statistics from /collection_path/detailed_doc_stats/ 
        so that we can get everything for the top 10,000 documents for 
        each query generated by Dirichlet language model method.
        """
        single_queries = Query(self.collection_path).get_queries_of_length(1)
        queries = {ele['num']:ele['title'] for ele in single_queries}
        cs = CollectionStats(self.collection_path)
        res = {}
        for qid in queries:
            res[qid] = []
            idx = 0
            for row in cs.get_qid_details(qid):
                docid = row['docid']
                doclen = float(row['doc_len'])
                #score = self.hypothesis_tf_function(tf, _type, scale, mu, sigma)
                res[qid].append([docid, doclen])
                idx += 1
                if idx >= 1000:
                    break
        return res

    def cal_tf_score(self, tfs, _min, _max, paras):
        for qid in tfs:
            for ele in tfs[qid]:
                if ele[1] > _min and ele[1] <= _max:
                    ele[1] = paras[2]+scipy.stats.norm(paras[0], paras[1]).pdf(ele[1])

    def outupt_tf_score(self, tfs):
        with open(os.path.join(self.collection_path, 'merged_results', 'title-method:hypothesis_stq_tf_2'), 'wb') as output:
            for qid in tfs:
                tfs[qid].sort(key=itemgetter(1), reverse=True)
                for ele in tfs[qid]:
                    output.write('%s Q0 %s 1 %s 0\n' % (qid, ele[0], ele[1]))

    def eval_tf_score(self):
        qrel_program='trec_eval -m all_trec -q'.split()
        result_file_path=os.path.join(self.collection_path, 'merged_results', 'title-method:hypothesis_stq_tf_2')
        eval_file_path=os.path.join(self.collection_path, 'evals', 'title-method:hypothesis_stq_tf_2')
        qrel_path=os.path.join(self.collection_path, 'judgement_file')
        qrel_program.append(qrel_path)
        qrel_program.append(result_file_path)
        #print qrel_program
        process = Popen(qrel_program, stdout=PIPE)
        stdout, stderr = process.communicate()
        all_performances = {}
        for line in stdout.split('\n'):
            line = line.strip()
            if line:
                row = line.split()
                evaluation_method = row[0]
                qid = row[1]
                try:
                    performace = ast.literal_eval(row[2])
                except:
                    continue

                if qid not in all_performances:
                    all_performances[qid] = {}
                all_performances[qid][evaluation_method] = performace

        with open( eval_file_path, 'wb' ) as o:
            json.dump(all_performances, o, indent=2)

    def plot_hypothesis_tf_curve_fit(self, ax, xaxis, yaxis):
        tfs = self.get_docs_tf()
        func = self.hypothesis_tf_function
        x10 = []
        y10 = []
        for d in zip(xaxis, yaxis):
            if d[0] <= 10:
                x10.append(d[0])
                y10.append(d[1])
        popt, pcov = curve_fit(func, x10, y10, method='trf', bounds=([0., -np.inf, 0], [10., np.inf, 1]))
        print popt
        self.cal_tf_score(tfs, 0, 10, popt)
        trialX = np.linspace(0, 10, 100)
        trialY = func(trialX, *popt)
        #print trialX, trialY
        #raw_input()
        ax.plot(trialX, trialY, label='fitting')

        x20 = []
        y20 = []
        for d in zip(xaxis, yaxis):
            if d[0] > 10 and d[0] <= 20:
                x20.append(d[0])
                y20.append(d[1])
        #print x20, y20
        popt, pcov = curve_fit(func, x20, y20, method='trf', bounds=([10., -np.inf, 0], [20., np.inf, 1]))
        print popt
        self.cal_tf_score(tfs, 10, 20, popt)
        trialX = np.linspace(10, 20, 100)
        trialY = func(trialX, *popt)
        # print trialX, trialY
        # raw_input()
        ax.plot(trialX, trialY, label='fitting')
        ax.legend(loc='best')

        self.outupt_tf_score(tfs)
        self.eval_tf_score()


    def hypothesis_tfln_function(self, tfln, x, y):
        return y+np.power(tfln, x)

    def plot_hypothesis_tfln_curve_fit(self, ax, xaxis, yaxis):
        tfs = self.get_docs_tf()
        lens = self.get_docs_len()
        func = self.hypothesis_tfln_function
        popt, pcov = curve_fit(func, xaxis, yaxis, method='trf')
        print popt
        maxX = np.max(xaxis)
        trialX = np.linspace(0, maxX, 100)
        trialY = func(trialX, *popt)
        #print trialX, trialY
        #raw_input()
        ax.plot(trialX, trialY, label='fitting')


    def mixture_exponential(self, xaxis, pi, l1, l2):
        return pi*scipy.stats.expon(l1).pdf(xaxis) + (1-pi)*scipy.stats.expon(l2).pdf(xaxis)

    def cal_curve_fit(self, ax, xaxis, yaxis, mode=1, paras=[]):
        if mode == 1:
            func = self.mixture_exponential
        popt, pcov = curve_fit(func, xaxis, yaxis, method='lm', p0=paras)
        print popt
        trialX = xaxis
        trialY = func(xaxis, *popt)
        return trialX, trialY

    def plot_single_tfc_constraints_rel_tf(self, query_length, x_func, 
            _method, plot_ratio=True, plot_total_or_avg=True,
            plot_rel_or_all=True, performance_as_legend=True, 
            drawline=True, numbins=60, xlimit=0, 
            ylimit=0, zoom_x=0, compact_x=False, oformat='eps'):
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
        @compact_x: map the x to 1,2,3,4,....
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
        #print np.mean([len(rel_docs[qid]) for qid in rel_docs])
        eval_class = Evaluation(self.collection_path)
        p = eval_class.get_all_performance_of_some_queries(
            method=_method,
            qids=queries.keys(), 
            return_all_metrics=False, 
            metrics=['map']
        )
        collection_x_dict = {}
        collection_level_maxTF = 0
        collection_level_maxX = 0.0
        num_cols = 4
        num_rows = int(math.ceil(len(rel_docs)*1.0/num_cols))
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, sharex=False, sharey=False, figsize=(0.5*num_cols, 0.5*num_rows))
        font = {'size' : 8}
        plt.rc('font', **font)
        row_idx = 0
        col_idx = 0
        idfs = [(qid, math.log(cs.get_term_IDF1(queries[qid]))) for qid in rel_docs]
        idfs.sort(key=itemgetter(1))
        all_expected_maps = []
        for qid,idf in idfs:
            if num_rows > 1:
                ax = axs[row_idx][col_idx]
            else:
                ax = axs[col_idx]
            col_idx += 1
            if col_idx >= num_cols:
                row_idx += 1
                col_idx = 0
            query_term = queries[qid]
            maxTF = cs.get_term_maxTF(query_term)
            #idf = math.log(cs.get_term_IDF1(query_term))
            legend = 'idf:%.2f'%idf
            if performance_as_legend:
                legend += '\nmap:%.4f' % (p[qid]['map'] if p[qid] else 0)
            if maxTF > collection_level_maxTF:
                collection_level_maxTF = maxTF
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
            yaxis = [x_dict[x][0] for x in xaxis]
            ranking_list = [(x_dict[x][0], x_dict[x][1]) for x in xaxis]
            all_expected_maps.append(self.cal_expected_map(ranking_list))
            if plot_ratio:
                yaxis = [x_dict[x][0]*1.0/x_dict[x][1] for x in xaxis]
            else:
                yaxis = [x_dict[x][0] for x in xaxis]
            #print xaxis
            p1 = np.asarray([x_dict[x][1] for x in xaxis])
            p2 = scipy.stats.poisson.pmf(xaxis, 4)
            # print xaxis
            # print p1
            # print p2
            # print p1*p2
            # print np.asarray(yaxis)
            # raw_input()
            query_stat = cs.get_term_stats(query_term)
            if drawline:
                self.plot_single_tfc_constraints_draw_pdf_line(
                    ax, xaxis, yaxis,
                    qid+'-'+query_term, 
                    legend,
                    True,
                    xlog=False,
                    legend_pos='best', 
                    xlabel_format=1,
                    xlimit=xlimit,
                    ylimit=ylimit)
            else:
                self.plot_single_tfc_constraints_draw_pdf_dot(
                    ax, xaxis, yaxis,
                    qid+'-'+query_term, 
                    legend,
                    True,
                    xlog=False,
                    legend_pos='best', 
                    xlabel_format=1,
                    xlimit=xlimit,
                    ylimit=ylimit)
        output_fn = os.path.join(self.all_results_root, output_root, 
            '%s-%s-%s-%s-%s-%s-%d-%.1f-%.1f-individual.%s' % (
                collection_name, 
                _method, 
                'ratio' if plot_ratio else 'abscnt', 
                'total' if plot_total_or_avg else 'avg',
                'rel' if plot_rel_or_all else 'all',
                'line' if drawline else 'dots', 
                numbins, 
                xlimit,
                ylimit, 
                oformat) )
        #plt.savefig(output_fn, format=oformat, bbox_inches='tight', dpi=400)

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
                yaxis = [(collection_x_dict[x][0]/len(idfs)) if plot_rel_or_all else (collection_x_dict[x][1]/len(idfs)) for x in xaxis]
            print np.sum(yaxis[20:]), np.sum(yaxis[20:])
        if numbins > 0:
            interval = collection_level_maxX*1.0/numbins
            newxaxis = [i for i in np.arange(0, collection_level_maxX+1e-10, interval)]
            newyaxis = [[0.0, 0.0] for x in newxaxis]
            for x in xaxis:
                newx = int(x / interval)
                newyaxis[newx][0] += collection_x_dict[x][0]
                newyaxis[newx][1] += collection_x_dict[x][1]
                # print x, newx
                # print newxaxis
                # print newyaxis
                # raw_input()
            xaxis = newxaxis
            if plot_ratio:
                yaxis = [ele[0]/ele[1] if ele[1] != 0 else 0.0 for ele in newyaxis]
            else:
                if plot_total_or_avg:
                    yaxis = [(ele[0]) if plot_rel_or_all else (ele[1]) for ele in newyaxis] 
                else:
                    yaxis = [(ele[0]/len(idfs)) if plot_rel_or_all else (ele[1]/len(idfs)) for ele in newyaxis]

        # we do not care about the actual values of x
        # so we just map the actual values to integer values

        return_data = copy.deepcopy(collection_x_dict)
        if compact_x:
            xaxis = range(1, len(xaxis)+1)

        collection_legend = ''
        if performance_as_legend:
            collection_legend = '$MAP:%.4f$' % (np.mean([p[qid]['map'] if p[qid] else 0 for qid in queries]))
            collection_legend += '\n$MAP_E:%.4f$' % (np.mean(all_expected_maps))

        zoom_xaxis = xaxis[zoom_x:]
        zoom_yaxis = yaxis[zoom_x:]
        if drawline:
            self.plot_single_tfc_constraints_draw_pdf_line(axs, xaxis, 
                yaxis, collection_name, 
                collection_legend, 
                legend_pos='best',
                xlimit=xlimit,
                ylimit=ylimit,
                zoom=zoom_x > 0,
                zoom_xaxis=zoom_xaxis,
                zoom_yaxis=zoom_yaxis)
            # only if we want to draw the fitting curve
            """
            if plotbins:
                self.plot_hypothesis_tfln_curve_fit(axs, xaxis, yaxis)
            """
        else:
            self.plot_single_tfc_constraints_draw_pdf_dot(axs, xaxis, 
                yaxis, collection_name, 
                collection_legend, 
                legend_pos='best',
                xlimit=xlimit,
                ylimit=ylimit,
                zoom=zoom_x > 0,
                zoom_xaxis=zoom_xaxis,
                zoom_yaxis=zoom_yaxis)

        output_fn = os.path.join(self.all_results_root, output_root, 
            '%s-%s-%s-%s-%s-%s-%d-%.1f-%.1f-all.%s' % (
                collection_name, 
                _method, 
                'ratio' if plot_ratio else 'abscnt', 
                'total' if plot_total_or_avg else 'avg',
                'rel' if plot_rel_or_all else 'all',
                'line' if drawline else 'dots', 
                numbins, 
                xlimit,
                ylimit, 
                oformat) )
        plt.savefig(output_fn, format=oformat, bbox_inches='tight', dpi=400)

        return collection_name, return_data, legend, _method

    def okapi(self, collection_stats, row):
        """
        okapi
        """
        return round((1.2+1)*float(row['total_tf'])/(float(row['total_tf']) + 1.2*(1-0.+0.*float(row['doc_len'])/float(collection_stats.get_avdl()))), 3) 
    def tf_1(self, collection_stats, row):
        """
        tf
        """
        return int(row['total_tf'])
    def tf_4(self, collection_stats, row):
        """
        1+log(1+log(tf))
        """
        return round(1+math.log(1+math.log(int(row['total_tf']))), 3)
    def tf_5(self, collection_stats, row):
        """
        tf/(tf+k)  k=1.0 default
        """
        return round(int(row['total_tf']) / (1.0 + int(row['total_tf'])), 4)
    def tfidf1(self, collection_stats, row):
        """
        tf/(tf+k) * idf  k=1.0 default
        """
        return round(int(row['total_tf']) / (1.0 + int(row['total_tf'])), 4)
    def tf_dl_1(self, collection_stats, row):
        """
        tf/dl
        """
        return round(float(row['total_tf'])/float(row['doc_len']), 3) 

    def tf_dl_3(self, collection_stats, row):
        """
        log(tf)/(tf+log(dl))
        """
        return round(np.log(float(row['total_tf']))/(float(row['total_tf'])+np.log(float(row['doc_len']))), 3) 
    def tf_dl_5(self, collection_stats, row):
        """
        (log(tf)+delta)/(tf+log(dl))
        """
        delta = 2.75
        return round((np.log(float(row['total_tf']))+delta)/np.log(float(row['doc_len'])), 3) 

    def wrapper(self, query_length, method_name, plot_ratio, plot_total_or_avg, 
            plot_rel_or_all, performance_as_legend, drawline,
            numbins, xlimit, ylimit, zoom_x, compact_x=False, oformat='eps'):
        """
        This is the wrapper of the actual function. 
        We parse the CLI arguments and convert them to the values required 
        for the function.
        """
        formal_method_name = method_name
        if method_name == 'okapi':
            x_func = self.okapi
            formal_method_name = 'okapi,b:0.0'
        elif method_name == 'tf_1':
            x_func = self.tf_1
            formal_method_name = 'tf1'
        elif method_name == 'tf_4':
            x_func = self.tf_4
            formal_method_name = 'tf4'
        elif method_name == 'tf_5':
            x_func = self.tf_5
            formal_method_name = 'tf5'
        elif method_name == 'tf_ln_1':
            x_func = self.tf_dl_1
            formal_method_name = 'hypothesis_stq_tf_ln_1'
        elif method_name == 'tf_ln_3':
            x_func = self.tf_dl_3
            formal_method_name = 'hypothesis_stq_tf_ln_3'
        elif method_name == 'tf_ln_5':
            x_func = self.tf_dl_5
            formal_method_name = 'hypothesis_stq_tf_ln_5'
        return self.plot_single_tfc_constraints_rel_tf(
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
            oformat
        )

    def plot_with_data_single(self, xaxis, yaxis, title, legend, xlabel, ylabel, 
            output_fn, query_length, method_name, plot_ratio, 
            plot_total_or_avg, plot_rel_or_all, performance_as_legend, 
            drawline, numbins, xlimit, ylimit, zoom_x=20, oformat='eps'):
        fig, axs = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(6, 3.*1))
        font = {'size' : 12}
        plt.rc('font', **font)
        zoom_xaxis = xaxis[zoom_x:]
        zoom_yaxis = yaxis[zoom_x:]
        if drawline:
            self.plot_single_tfc_constraints_draw_pdf_line(
                axs, 
                xaxis, 
                yaxis, 
                title, 
                legend, 
                xlabel=xlabel,
                ylabel=ylabel,
                zoom=zoom_x > 0,
                zoom_xaxis=zoom_xaxis,
                zoom_yaxis=zoom_yaxis,
                legend_pos='best',
                xlimit=xlimit,
                ylimit=ylimit)
        else:
            self.plot_single_tfc_constraints_draw_pdf_dot(
                axs, 
                xaxis, 
                yaxis, 
                title, 
                legend, 
                xlabel=xlabel,
                ylabel=ylabel,
                zoom=zoom_x > 0,
                zoom_xaxis=zoom_xaxis,
                zoom_yaxis=zoom_yaxis,
                legend_pos='best',
                xlimit=xlimit,
                ylimit=ylimit)

        plt.savefig(output_fn, format=oformat, bbox_inches='tight', dpi=400)

    def plot_with_data(self, data, title, legend, 
            query_length, method_name, plot_ratio, 
            plot_total_or_avg, plot_rel_or_all, performance_as_legend, 
            drawline, numbins, xlimit, ylimit, zoom_x=20, 
            compact_x=False, oformat='eps'):
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

        xaxis = sorted(data.keys())
        yaxis = [[data[x][0]*1./data[x][1] for x in xaxis], [data[x][0] for x in xaxis], [data[x][1] for x in xaxis]]
        ylabels = ['ratio of rel docs', 'rel docs count', 'docs count']

        sum_rel = sum([data[x][0] for x in xaxis])
        sum_all = sum([data[x][1] for x in xaxis])
        y_prob = [[data[x][0]*1.0/sum_rel for x in xaxis], [data[x][1]*1.0/sum_all for x in xaxis]]
        # y_fitting_paras = [EM().exponential(ele) for ele in y_prob]
        # y_fitting = [[y_fitting_paras[0][0][0]*math.exp(-x*y_fitting_paras[0][0][0])*y_fitting_paras[0][1][0]+y_fitting_paras[0][0][1]*math.exp(-x*y_fitting_paras[0][0][1])*y_fitting_paras[0][1][1] for x in xaxis], 
        #     [y_fitting_paras[1][0][0]*math.exp(-x*y_fitting_paras[1][0][0])*y_fitting_paras[1][1][0]+y_fitting_paras[1][0][1]*math.exp(-x*y_fitting_paras[1][0][1])*y_fitting_paras[1][1][1] for x in xaxis]]
        # print y_fitting
        
        yprob_labels = ['PDF of rel docs', 'PDF of docs']
        
        if compact_x:
            xaxis = range(1, len(xaxis)+1)

        for i, ele in enumerate(y_prob):
            y_fitting = self.cal_curve_fit(None, xaxis, ele, 1, [0.5, 1, 1])
            y_prob[i].append(y_fitting[1])

        output_root = os.path.join('collection_figures', query_length)
        if not os.path.exists(os.path.join(self.all_results_root, output_root)):
            os.makedirs(os.path.join(self.all_results_root, output_root))
        for i in range(3):
            output_fn = os.path.join(self.all_results_root, output_root, 
                '%s-%s-%s-%s-%s-%s-%d-%.1f-%.1f.%s' % (
                    'all_collections', 
                    method_name, 
                    'ratio' if i==0 else 'abscnt', 
                    'total',
                    'rel' if i==1 else 'all',
                    'line' if drawline else 'dots', 
                    numbins, 
                    xlimit,
                    ylimit, 
                    oformat) )
            self.plot_with_data_single(xaxis, yaxis[i], title, legend, 'projected doc score', 
                ylabels[i], output_fn, query_length, method_name, plot_ratio, 
                plot_total_or_avg, plot_rel_or_all, performance_as_legend, 
                drawline, numbins, xlimit, ylimit, zoom_x, oformat)
        for i in range(2):
            output_fn = os.path.join(self.all_results_root, output_root, 
                '%s-%s-%s-%s-%s-%d-%.1f-%.1f.%s' % (
                    'all_collections', 
                    method_name, 
                    'rel_dist' if i==0 else 'all_dist', 
                    'total',
                    'line' if drawline else 'dots', 
                    numbins, 
                    xlimit,
                    ylimit, 
                    oformat) )
            self.plot_with_data_single(xaxis, y_prob[i], title, legend, 'projected doc score', 
                yprob_labels[i], output_fn, query_length, method_name, plot_ratio, 
                plot_total_or_avg, plot_rel_or_all, performance_as_legend, 
                drawline, numbins, xlimit, ylimit, zoom_x, oformat)