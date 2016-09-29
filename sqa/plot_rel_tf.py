# -*- coding: utf-8 -*-
import sys,os
import math
import argparse
import json
import ast
from operator import itemgetter
from subprocess import Popen, PIPE

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils/'))
from base import SingleQueryAnalysis
from collection_stats import CollectionStats
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


class PlotRelTF(SingleQueryAnalysis):
    """
    Plot the probability distribution of P(D=1|TF=x)
    """
    def __init__(self):
        super(PlotRelTF, self).__init__()
        self.collection_path = ''

    def plot_single_tfc_constraints_draw_pdf(self, ax, xaxis, yaxis, 
            title, legend, legend_outside=False, marker='ro', 
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


    def plot_single_tfc_constraints_draw_pdf_line(self, ax, xaxis, yaxis, 
            title, legend, legend_outside=False, marker=None, linestyle=None, 
            xlog=True, ylog=False, zoom=False, legend_pos='upper right', 
            xlabel_format=0):
        # 1. probability distribution 
        ax.plot(xaxis, yaxis, marker=marker if marker else '+', ls=linestyle if linestyle else '-', label=legend)
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

    def plot_single_tfc_constraints_draw_kde(self, ax, yaxis, _bandwidth=0.5):
        # kernel density estimation 
        #print '_bandwidth:'+str(_bandwidth)
        yaxis = np.asarray(yaxis)[:, np.newaxis]
        kde = KernelDensity(kernel='gaussian', bandwidth=_bandwidth).fit(yaxis)
        X_plot = np.linspace(0, len(yaxis), 100)[:, np.newaxis]
        log_dens = kde.score_samples(X_plot)
        ax.fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF',
                label='KDE')
        #print kde
        ax.legend(loc='best')

    def plot_single_tfc_constraints_draw_hist(self, ax, yaxis, nbins, _norm, title, legend):
        #2. hist gram
        yaxis.sort()
        n, bins, patches = ax.hist(yaxis, nbins, normed=_norm, facecolor='#F08080', alpha=0.5, label=legend)
        ax.set_title(title)
        ax.legend()


    def plot_single_tfc_constraints_tf_rel(self, collection_path, smoothing=True, oformat='eps'):
        collection_name = collection_path.split('/')[-1]
        cs = CollectionStats(collection_path)
        output_root = 'single_query_figures'
        single_queries = Query(collection_path).get_queries_of_length(1)
        queries = {ele['num']:ele['title'] for ele in single_queries}
        #print queries
        rel_docs = Judgment(collection_path).get_relevant_docs_of_some_queries(queries.keys(), 1, 'dict')
        #print rel_docs
        #raw_input()
        collection_level_tfs = []
        collection_level_x_dict = {}
        collection_level_maxTF = 0
        collection_level_maxTFLN = 0.0
        num_cols = 4
        num_rows = int(math.ceil(len(rel_docs)*1.0/num_cols*2))
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, sharex=False, sharey=False, figsize=(3*num_cols, 3.*num_rows))
        font = {'size' : 8}
        plt.rc('font', **font)
        row_idx = 0
        col_idx = 0
        for qid in sorted(rel_docs):
            ax1 = axs[row_idx][col_idx]
            ax2 = axs[row_idx][col_idx+1]
            col_idx += 2
            if col_idx >= num_cols:
                row_idx += 1
                col_idx = 0
            query_term = queries[qid]
            maxTF = cs.get_term_maxTF(query_term)
            if maxTF > collection_level_maxTF:
                collection_level_maxTF = maxTF
            #print maxTF
            idf = cs.get_term_IDF1(query_term)
            tfs = [int(tf) for tf in cs.get_term_docs_tf_of_term_with_qid(qid, query_term, rel_docs[qid].keys())]
            rel_docs_len = len( rel_docs[qid].keys() )
            #print tfs, rel_docs_len
            doc_with_zero_tf_len = len( rel_docs[qid].keys() ) - len(tfs)
            tfs.extend([0]*doc_with_zero_tf_len)
            collection_level_tfs.extend(tfs)
            #print len( rel_docs[qid].keys() )
            #print len(tfs)
            x_dict = {}
            for tf in tfs:
                if tf not in x_dict:
                    x_dict[tf] = 0
                x_dict[tf] += 1
                if tf not in collection_level_x_dict:
                    collection_level_x_dict[tf] = 0
                collection_level_x_dict[tf] += 1
            x_dict[0] = len( rel_docs[qid].keys() ) - len(tfs)
            yaxis_hist = tfs
            yaxis_hist.sort()
            #print len(yaxis_hist)
            #print yaxis_hist
            yaxis_all = []
            for tf in range(0, maxTF+1):
                if tf not in x_dict:
                    x_dict[tf] = 0
                else:
                    yaxis_all.extend([tf+.1]*x_dict[tf])
                if smoothing:
                    x_dict[tf] += .1
                    rel_docs_len += .1

                if tf not in collection_level_x_dict:
                    collection_level_x_dict[tf] = 0
                if smoothing:
                    collection_level_x_dict[tf] += .1

            xaxis = x_dict.keys()
            xaxis.sort()
            yaxis_pdf = [x_dict[x]/rel_docs_len for x in xaxis]

            self.plot_single_tfc_constraints_draw_pdf(ax1, xaxis, 
                yaxis_pdf, qid+'-'+query_term, 
                "maxTF=%d\n|rel_docs|=%d\nidf=%.1f" % (maxTF, rel_docs_len, idf), 
                xlog=False)
            self.plot_single_tfc_constraints_draw_kde(ax1, yaxis_all, 1.06*math.pow(len(yaxis_all), -0.2)*np.std(yaxis_all))
            self.plot_single_tfc_constraints_draw_hist(ax2, yaxis_hist, 
                math.ceil(maxTF/10.), False, qid+'-'+query_term, 
                '#bins(maxTF/10.0)=%d' % (math.ceil(maxTF/10.)))

        fig.text(0.5, 0.07, 'Term Frequency', ha='center', va='center', fontsize=12)
        fig.text(0.06, 0.5, 'P( c(t,D)=x | D is a relevant document)=tf/|rel_docs|', ha='center', va='center', rotation='vertical', fontsize=12)
        fig.text(0.5, 0.5, 'Histgram', ha='center', va='center', rotation='vertical', fontsize=12)
        fig.text(0.6, 0.04, 'Histgram:rel docs are binned by their TFs. The length of the bin is set to 10. Y axis shows the number of rel docs in each bin.', ha='center', va='center', fontsize=10)

        plt.savefig(os.path.join(self.all_results_root, output_root, collection_name+'-tf_rel.'+oformat), 
            format=oformat, bbox_inches='tight', dpi=400)

        #collection level
        collection_level_xaxis = collection_level_x_dict.keys()
        collection_level_xaxis.sort()
        collection_level_yaxis_pdf = [collection_level_x_dict[x]/len(collection_level_tfs) for x in collection_level_xaxis]

        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False, figsize=(6, 2.*2))
        font = {'size' : 8}
        plt.rc('font', **font)
        self.plot_single_tfc_constraints_draw_pdf(axs[0], collection_level_xaxis, 
            collection_level_yaxis_pdf, collection_name, "", ylog=False)
        self.plot_single_tfc_constraints_draw_hist(axs[1], collection_level_tfs, 
            math.ceil(collection_level_maxTF/10.), False, "", 
            '#bins(collection_level_maxTF/10.0)=%d' % (math.ceil(collection_level_maxTF/10.)))
        #fig.text(0.5, 0.07, 'Term Frequency', ha='center', va='center', fontsize=12)
        #fig.text(0.06, 0.5, 'P( c(t,D)=x | D is a relevant document)=tf/|rel_docs|', ha='center', va='center', rotation='vertical', fontsize=12)
        #fig.text(0.5, 0.5, 'Histgram', ha='center', va='center', rotation='vertical', fontsize=12)
        #fig.text(0.6, 0.04, 'Histgram:rel docs are binned by their TFs. The length of the bin is set to 10. Y axis shows the number of rel docs in each bin.', ha='center', va='center', fontsize=10)

        plt.savefig(os.path.join(self.all_results_root, output_root, collection_name+'-all-tf_rel.'+oformat), 
            format=oformat, bbox_inches='tight', dpi=400)



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


    def plot_single_tfc_constraints_rel_tf(self, plot_ratio=True, 
            performance_as_legend=True, plot_tf_ln=True, 
            drawline=True, plotbins=True, numbins=60, 
            smoothing=False, oformat='eps'):
        collection_name = self.collection_path.split('/')[-1]
        print collection_name
        cs = CollectionStats(self.collection_path)
        output_root = 'single_query_figures'
        single_queries = Query(self.collection_path).get_queries_of_length(1)
        queries = {ele['num']:ele['title'] for ele in single_queries}
        #print qids
        rel_docs = Judgment(self.collection_path).get_relevant_docs_of_some_queries(queries.keys(), 1, 'dict')
        #print rel_docs
        collection_legend = ''
        if performance_as_legend:
            baseline_performances = Evaluation(self.collection_path).get_all_performance_of_some_queries(
                qids=queries.keys(), 
                return_all_metrics=False, 
                metrics=['map']
            )
            optimals = Performances(self.collection_path).load_optimal_performance(['dir', 'okapi'])
            dirichlet_performances = Evaluation(self.collection_path).get_all_performance_of_some_queries(
                method=optimals[0][0]+','+optimals[0][2],
                qids=queries.keys(), 
                return_all_metrics=False, 
                metrics=['map']
            )
            okapi_performances = Evaluation(self.collection_path).get_all_performance_of_some_queries(
                method=optimals[1][0]+','+optimals[1][2],
                qids=queries.keys(), 
                return_all_metrics=False, 
                metrics=['map']
            )
            tf_performances = Evaluation(self.collection_path).get_all_performance_of_some_queries(
                method='tf1',
                qids=queries.keys(), 
                return_all_metrics=False, 
                metrics=['map']
            )
            hypothesis_tf1_performances = Evaluation(self.collection_path).get_all_performance_of_some_queries(
                method='hypothesis_stq_tf',
                qids=queries.keys(), 
                return_all_metrics=False, 
                metrics=['map']
            )
            try:
                hypothesis_tf2_performances = Evaluation(self.collection_path).get_all_performance_of_some_queries(
                    method='hypothesis_stq_tf_2',
                    qids=queries.keys(), 
                    return_all_metrics=False, 
                    metrics=['map']
                )
            except:
                pass
            try:
                hypothesis_tf_ln_1_performances = Evaluation(self.collection_path).get_all_performance_of_some_queries(
                    method='hypothesis_stq_tf_ln_1',
                    qids=queries.keys(), 
                    return_all_metrics=False, 
                    metrics=['map']
                )
            except:
                pass
            try:
                hypothesis_tf_ln_3_performances = Evaluation(self.collection_path).get_all_performance_of_some_queries(
                    method='hypothesis_stq_tf_ln_3',
                    qids=queries.keys(), 
                    return_all_metrics=False, 
                    metrics=['map']
                )
            except:
                pass
            try:
                hypothesis_tf_ln_5_performances = Evaluation(self.collection_path).get_all_performance_of_some_queries(
                    method='hypothesis_stq_tf_ln_5',
                    qids=queries.keys(), 
                    return_all_metrics=False, 
                    metrics=['map']
                )
            except:
                pass

            avg_baseline_performance = np.mean([baseline_performances[qid]['map'] for qid in baseline_performances])
            #legend += 'avg_map(lm):%.3f' % avg_baseline_performance
            avg_dirichlet_performance = np.mean([dirichlet_performances[qid]['map'] for qid in dirichlet_performances])
            collection_legend += 'avg_map(dir):%.4f' % avg_dirichlet_performance
            avg_okapi_performance = np.mean([okapi_performances[qid]['map'] for qid in okapi_performances])
            collection_legend += '\navg_map(okapi):%.4f' % avg_okapi_performance
            avg_tf_performances = np.mean([tf_performances[qid]['map'] for qid in tf_performances])
            collection_legend += '\navg_map(tf):%.4f' % avg_tf_performances
            avg_hypothesis_tf1_performances = np.mean([hypothesis_tf1_performances[qid]['map'] for qid in hypothesis_tf1_performances])
            #collection_legend += '\navg_map(hypo_tf1):%.3f' % avg_hypothesis_tf1_performances
            try:
                avg_hypothesis_tf2_performances = np.mean([hypothesis_tf2_performances[qid]['map'] for qid in hypothesis_tf2_performances])
                #collection_legend += '\navg_map(hypo_tf2):%.3f' % avg_hypothesis_tf2_performances
            except:
                pass
            try:
                avg_hypothesis_tf_ln_1_performances = np.mean([hypothesis_tf_ln_1_performances[qid]['map'] for qid in hypothesis_tf_ln_1_performances])
                collection_legend += '\navg_map(tf_ln_1):%.4f' % avg_hypothesis_tf_ln_1_performances
            except:
                pass
            try:
                avg_hypothesis_tf_ln_3_performances = np.mean([hypothesis_tf_ln_3_performances[qid]['map'] for qid in hypothesis_tf_ln_3_performances])
                collection_legend += '\navg_map(tf_ln_3):%.4f' % avg_hypothesis_tf_ln_3_performances
            except:
                pass
            try:
                avg_hypothesis_tf_ln_5_performances = np.mean([hypothesis_tf_ln_5_performances[qid]['map'] for qid in hypothesis_tf_ln_5_performances])
                collection_legend += '\navg_map(tf_ln_5):%.4f' % avg_hypothesis_tf_ln_5_performances
            except:
                pass

        collection_level_x_dict = {}
        collection_level_maxTF = 0
        collection_level_maxX = 0.0
        num_cols = 4
        num_rows = int(math.ceil(len(rel_docs)*1.0/num_cols))
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, sharex=False, sharey=False, figsize=(3*num_cols, 3.*num_rows))
        font = {'size' : 8}
        plt.rc('font', **font)
        row_idx = 0
        col_idx = 0
        idfs = [(qid, math.log(cs.get_term_IDF1(queries[qid]))) for qid in rel_docs]
        idfs.sort(key=itemgetter(1))
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
                baseline_performance = baseline_performances[qid]['map']
                #legend += '\nmap(lm):%.3f' % baseline_performance
                dirichlet_performance = dirichlet_performances[qid]['map']
                legend += '\nmap(dir):%.4f' % dirichlet_performance
                okapi_performance = okapi_performances[qid]['map']
                legend += '\nmap(okapi):%.4f' % okapi_performance
                tf_performance = tf_performances[qid]['map']
                legend += '\nmap(tf):%.4f' % tf_performance
                hypothesis_tf1_performance = hypothesis_tf1_performances[qid]['map']
                #legend += '\nmap(hypo_tf1):%.3f' % hypothesis_tf1_performance
                try:
                    hypothesis_tf2_performance = hypothesis_tf2_performances[qid]['map']
                    #legend += '\nmap(hypo_tf2):%.3f' % hypothesis_tf2_performance
                except:
                    pass
                try:
                    hypothesis_tf_ln_1_performance = hypothesis_tf_ln_1_performances[qid]['map']
                    legend += '\nmap(tf_ln_1):%.4f' % hypothesis_tf_ln_1_performance
                except:
                    pass
                try:
                    hypothesis_tf_ln_3_performance = hypothesis_tf_ln_3_performances[qid]['map']
                    legend += '\nmap(tf_ln_3):%.4f' % hypothesis_tf_ln_3_performance
                except:
                    pass
                try:
                    hypothesis_tf_ln_5_performance = hypothesis_tf_ln_5_performances[qid]['map']
                    legend += '\nmap(tf_ln_5):%.4f' % hypothesis_tf_ln_5_performance
                except:
                    pass
            if maxTF > collection_level_maxTF:
                collection_level_maxTF = maxTF
            #query_term_ivlist = cs.get_term_counts_dict(query_term)
            # detailed_stats_fn = os.path.join(collection_path, 'docs_statistics_json', qid)
            # with open(detailed_stats_fn) as f:
            #     detailed_stats_json = json.load(f)
            # tf_dict = {}
            # tf_dict = {k:[v['TOTAL_TF'], k in rel_docs[qid]] for k,v in detailed_stats_json.items()}
            # x_dict = {}
            # for docid, values in tf_dict.items():
            x_dict = {}
            qid_docs_len = 0
            yaxis_all = []
            for row in cs.get_qid_details(qid):
                qid_docs_len += 1
                if plot_tf_ln:
                    ## we use tf instead for code compatiablity

                    ##### tf_ln_3
                    tf = round(np.log(float(row['total_tf']))/(float(row['total_tf'])+np.log(float(row['doc_len']))), 3) 
                    ##### tf_ln_5
                    # if 'gov2' in self.collection_path:
                    #     delta = 3
                    # elif 'disk12' in self.collection_path:
                    #     delta = 0.05
                    # elif 'disk45' in self.collection_path:
                    #     delta = 0.0
                    # elif 'wt2g' in self.collection_path:
                    #     delta = 0.0
                    # tf = round((np.log(float(row['total_tf']))+delta)/np.log(float(row['doc_len'])), 3)
                else:
                    tf = int(row['total_tf'])
                if tf > collection_level_maxX:
                    collection_level_maxX = tf
                rel = (int(row['rel_score'])>=1)
                if tf not in x_dict:
                    x_dict[tf] = [0, 0] # [rel_docs, total_docs]
                if rel:
                    x_dict[tf][0] += 1
                x_dict[tf][1] += 1
                yaxis_all.append(tf)

                if tf not in collection_level_x_dict:
                    collection_level_x_dict[tf] = [0, 0] # [rel_docs, total_docs]
                if rel:
                    collection_level_x_dict[tf][0] += 1
                collection_level_x_dict[tf][1] += 1
            xaxis = x_dict.keys()
            xaxis.sort()
            yaxis = [x_dict[x][0] for x in xaxis]
            yaxis_total = [x_dict[x][1] for x in xaxis]
            yaxis_ratio = [x_dict[x][0]*1.0/x_dict[x][1] for x in xaxis]
            #print xaxis
            #print yaxis
            xaxis_splits_10 = [[x for x in xaxis if x <= i+10 and x > i] for i in range(0, maxTF+1, 10)]
            #print xaxis_splits_10
            yaxis_splits_10 = [[x_dict[x][0]*1./x_dict[x][1] for x in xaxis if x <= i+10 and x > i] for i in range(0, maxTF+1, 10)]
            #print yaxis_splits_10
            entropy_splits_10 = [entropy(ele, base=2) for ele in yaxis_splits_10]
            query_stat = cs.get_term_stats(query_term)
            dist_entropy = entropy(yaxis, base=2)
            if plot_ratio:
                self.plot_single_tfc_constraints_draw_pdf(ax, xaxis, 
                    yaxis_ratio, qid+'-'+query_term, 
                    legend,
                    True,
                    xlog=False,
                    legend_pos='best' if plot_tf_ln else 'upper right',
                    xlabel_format=1)
            else:
                self.plot_single_tfc_constraints_draw_pdf(ax, xaxis, yaxis_total, 
                    qid+'-'+query_term, 
                    'total\nidf:%.1f'%idf, 
                    True,
                    xlog=False)
                self.plot_single_tfc_constraints_draw_pdf(ax, xaxis, 
                    yaxis, qid+'-'+query_term, 
                    'rel', 
                    True,
                    marker='bs', 
                    xlog=False,
                    zoom=(qid =='349' or qid =='379' or qid =='392' or qid =='395' or qid =='417' or qid =='424'))


        collection_vocablulary_stat = cs.get_vocabulary_stats()
        collection_vocablulary_stat_str = ''
        idx = 1
        for k,v in collection_vocablulary_stat.items():
            collection_vocablulary_stat_str += k+'='+'%.2f'%v+' '
            if idx == 3:
                collection_vocablulary_stat_str += '\n'
                idx = 1
            idx += 1
        #fig.text(0.5, 0, collection_vocablulary_stat_str, ha='center', va='center', fontsize=12)
        if plot_ratio:
            if plot_tf_ln:
                output_fn = os.path.join(self.all_results_root, output_root, collection_name+'-rel_tf_ln_ratio.'+oformat)
            else:
                output_fn = os.path.join(self.all_results_root, output_root, collection_name+'-rel_tf_ratio.'+oformat)
        else:
            output_fn = os.path.join(self.all_results_root, output_root, collection_name+'-rel_tf.'+oformat)
        plt.savefig(output_fn, format=oformat, bbox_inches='tight', dpi=400)


        fig, axs = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(6, 3.*1))
        font = {'size' : 8}
        plt.rc('font', **font)
        xaxis = collection_level_x_dict.keys()
        xaxis.sort()
        yaxis = [collection_level_x_dict[x][0]*1./collection_level_x_dict[x][1] for x in xaxis]
        if plotbins:
            interval = collection_level_maxX*1.0/numbins
            newxaxis = [i for i in np.arange(0, collection_level_maxX+1e-10, interval)]
            newyaxis = [[0.0, 0.0] for x in newxaxis]
            for x in xaxis:
                newx = int(x / interval)
                newyaxis[newx][0] += collection_level_x_dict[x][0]
                newyaxis[newx][1] += collection_level_x_dict[x][1]
                # print x, newx
                # print newxaxis
                # print newyaxis
                # raw_input()
            xaxis = newxaxis
            yaxis = [ele[0]/ele[1] if ele[1] != 0 else 0.0 for ele in newyaxis]


        if drawline:
            self.plot_single_tfc_constraints_draw_pdf_line(axs, xaxis, 
                yaxis, collection_name, 
                collection_legend, 
                legend_pos='best' if plot_tf_ln else 'upper right',
                xlog=False,
                ylog=False)
            if plotbins:
                self.plot_hypothesis_tfln_curve_fit(axs, xaxis, yaxis)
        else:
            self.plot_single_tfc_constraints_draw_pdf(axs, xaxis, 
                yaxis, collection_name, 
                collection_legend, 
                legend_pos='best' if plot_tf_ln else 'upper right',
                xlog=False,
                ylog=False)
        if plot_tf_ln:
            if drawline:
                if plotbins:
                    plt.savefig(os.path.join(self.all_results_root, output_root, collection_name+'-alllinebins-rel_tf_ln_3.'+oformat), 
                        format=oformat, bbox_inches='tight', dpi=400)
                else:
                    plt.savefig(os.path.join(self.all_results_root, output_root, collection_name+'-allline-rel_tf_ln.'+oformat), 
                        format=oformat, bbox_inches='tight', dpi=400)
            else:
                if plotbins:
                    plt.savefig(os.path.join(self.all_results_root, output_root, collection_name+'-allbins-rel_tf_ln.'+oformat), 
                        format=oformat, bbox_inches='tight', dpi=400)
                else:
                    plt.savefig(os.path.join(self.all_results_root, output_root, collection_name+'-all-rel_tf_ln.'+oformat), 
                        format=oformat, bbox_inches='tight', dpi=400)
        else:
            if drawline:
                if plotbins:
                    plt.savefig(os.path.join(self.all_results_root, output_root, collection_name+'-alllinebins-rel_tf.'+oformat), 
                        format=oformat, bbox_inches='tight', dpi=400)
                else:
                    plt.savefig(os.path.join(self.all_results_root, output_root, collection_name+'-allline-rel_tf.'+oformat), 
                        format=oformat, bbox_inches='tight', dpi=400)
            else:
                if plotbins:
                    plt.savefig(os.path.join(self.all_results_root, output_root, collection_name+'-allbins-rel_tf.'+oformat), 
                        format=oformat, bbox_inches='tight', dpi=400)
                else:
                    plt.savefig(os.path.join(self.all_results_root, output_root, collection_name+'-all-rel_tf.'+oformat), 
                        format=oformat, bbox_inches='tight', dpi=400)
        return collection_name, xaxis, yaxis, legend, plot_tf_ln


    def plot_single_tfc_constraints(self):
        #self.plot_single_tfc_constraints_tf_rel(corpus_path)
        return self.plot_single_tfc_constraints_rel_tf()


    def plot_rel_tf_for_all_collections(self, data, smooth_curve=False, output_root='single_query_figures', oformat='eps'):
        """
        @input:
            -data: a list of data with each element as the data for a single collection 
                    with the format: (collection_name, xaxis, yaxis, legend)
        """
        fig, axs = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(6, 3.*1))
        font = {'size' : 8}
        plt.rc('font', **font)
        markers = ['ro', 'bs', 'kv', 'gx'] 
        for d in zip(data, markers):
            xaxis = d[0][1]
            yaxis = d[0][2]
            legend = d[0][0]
            if smooth_curve:
                x_smooth = np.linspace(xaxis[0], xaxis[-1], 2000)
                y_smooth = spline(xaxis, yaxis, x_smooth)
                xaxis = x_smooth
                yaxis = y_smooth
                axs.plot(xaxis, yaxis, d[1], label=legend)
                axs.set_xlim(0, axs.get_xlim()[1] if axs.get_xlim()[1]<100 else 100)
                axs.set_ylim(0, 1)
                axs.set_title('All Collections')
                axs.legend(loc='best')
            else:
                self.plot_single_tfc_constraints_draw_pdf(axs, xaxis, 
                    yaxis, 'All Collections', 
                    legend, 
                    marker='' if smooth_curve else d[1],
                    xlog=False,
                    ylog=False)
        box = axs.get_position()
        #axs.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # Put a legend to the right of the current axis
        #axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if data[0][4]: #plot_tf_ln
            plt.savefig(os.path.join(self.all_results_root, output_root, 'all-rel_tf_ln.'+oformat), 
                format=oformat, bbox_inches='tight', dpi=400)
        else:
            plt.savefig(os.path.join(self.all_results_root, output_root, 'all-rel_tf.'+oformat), 
                format=oformat, bbox_inches='tight', dpi=400)


    def plot_tfc_constraints(self, collections_path=[], smoothing=True):
        """
        * Start with the relevant document distribution according to one 
        term statistic and see how it affects the performance. 

        Take TF as an example and we start from the simplest case when 
        |Q|=1.  We could leverage an existing collection 
        and estimate P( c(t,D)=x | D is a relevant document), 
        where x = 0,1,2,...maxTF(t).

        Note that this step is function-independent.  We are looking at
        a general constraint for TF (i.e., TFC1), and to see how well 
        real collections would satisfy this constraint and then formalize
        the impact of TF on performance based on the rel doc distribution. 
        """
        print '-'*30
        print 'PLEASE RUN THIS PROGRAM TWICE IN ORDER TO GET THE CORRECT RESULTS!!!'
        print '-'*30
        all_rel_tf_data = []
        for c in collections_path:
            self.collection_path = os.path.abspath(c)
            cd = self.plot_single_tfc_constraints()
            all_rel_tf_data.append(cd)
        #self.plot_rel_tf_for_all_collections(all_rel_tf_data)

    def plot_tfc_constraints(self, collections_path=[], smoothing=True):
        """
        * Start with the relevant document distribution according to one 
        term statistic and see how it affects the performance. 

        Take TF as an example and we start from the simplest case when 
        |Q|=1.  We could leverage an existing collection 
        and estimate P( c(t,D)=x | D is a relevant document), 
        where x = 0,1,2,...maxTF(t).

        Note that this step is function-independent.  We are looking at
        a general constraint for TF (i.e., TFC1), and to see how well 
        real collections would satisfy this constraint and then formalize
        the impact of TF on performance based on the rel doc distribution. 
        """
        print '-'*30
        print 'PLEASE RUN THIS PROGRAM TWICE IN ORDER TO GET THE CORRECT RESULTS!!!'
        print '-'*30
        all_rel_tf_data = []
        for c in collections_path:
            self.collection_path = os.path.abspath(c)
            cd = self.plot_single_tfc_constraints()
            all_rel_tf_data.append(cd)
        #self.plot_rel_tf_for_all_collections(all_rel_tf_data)