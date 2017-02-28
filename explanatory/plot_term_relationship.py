# -*- coding: utf-8 -*-
import sys,os
import math
import argparse
import json
import csv
import ast
import copy
import re
import collections
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
from subqueries_learning import SubqueriesLearning

import numpy as np
import scipy.stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D


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

    def read_docdetails_data(self, query_length=2, only_rel=False):
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
            if only_rel:
                all_data[qid] = doc_details.get_only_rels(qid)
            else:
                all_data[qid] = doc_details.get_qid_details_as_numpy_arrays(qid)
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
        rel_contain_alls = {}
        for qid in details_data:
            terms = details_data[qid][0]
            all_tfs = details_data[qid][1]
            dfs = details_data[qid][2]
            doc_lens = details_data[qid][3]
            rels = details_data[qid][4]
            tf_in_docs = all_tfs.transpose()
            rel_mapped = []
            total_cnts = np.zeros(4)
            rel_contain_all = []
            for tf_idx, ele in enumerate(tf_in_docs):
                mapped = self.rel_mapping(ele, dfs)
                total_cnts[mapped] += 1
                if rels[tf_idx] > 0:
                    rel_mapped.append(mapped)
                    if mapped == 3: # if the doc contains all query terms
                        rel_contain_all.append(ele)
            rel_contain_alls[qid] = np.array(rel_contain_all)
            unique, counts = np.unique(rel_mapped, return_counts=True)
            countings[qid] = {value: {
                'cnt':counts[i], 
                'rel_ratio': counts[i]*1./rel_data[qid]['rel_cnt'],
                'total_ratio': counts[i]*1./total_cnts[value]} for i, value in enumerate(unique)}
            for i in range(1, 4):
                if i not in countings[qid]:
                    countings[qid][i] = {'cnt': 0, 'rel_ratio': 0, 'total_ratio': 0}
            if 0 not in countings[qid]:
                cnt = rel_data[qid]['rel_cnt'] - sum([countings[qid][v]['cnt'] for v in countings[qid]]) if qid in rel_data else 0
                countings[qid][0] = {'cnt': cnt, 'rel_ratio':cnt*1./rel_data[qid]['rel_cnt'] if qid in rel_data else 0, 'total_ratio': 0}
        return countings, rel_contain_alls


    def plot_all_kinds_of_docs(self, data, details_data, rel_data, query_length=2, oformat='png'):
        row_labels = ['cnt', 'rel_ratio', 'total_ratio']
        all_xaxis = np.array([[[data[qid][i][t] for qid in details_data] for i in range(4)] for t in row_labels])
        yaxis = [float(rel_data[qid]['AP']['okapi'][1]) for qid in rel_data] # yaxis is the performance, e.g. AP
        num_rows, num_cols = all_xaxis.shape[:2]
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, sharex=False, sharey=True, figsize=(3*num_cols, 3*num_rows))
        font = {'size' : 10}
        plt.rc('font', **font)
        row_idx = 0
        labels = ['NONE', 'LIDF', 'HIDF', 'ALL']
        markers = ['*', 's', '^', 'o']
        colors = ['k', 'r', 'g', 'b']
        for row_idx, ele in enumerate(all_xaxis):
            for col_idx, xaxis in enumerate(ele):
                ax = axs[row_idx][col_idx]
                zipped = zip(details_data.keys(), xaxis, yaxis)
                zipped.sort(key=itemgetter(1))
                qids_plot = np.array(zip(*zipped)[0])
                xaxis_plot = np.array(zip(*zipped)[1])
                yaxis_plot = np.array(zip(*zipped)[2])
                legend = 'pearsonr:%.4f' % (scipy.stats.pearsonr(xaxis_plot, yaxis_plot)[0])
                ax.plot(xaxis_plot, yaxis_plot, marker=markers[col_idx], mfc=colors[col_idx], ms=4, ls='None', label=legend)
                ax.set_title(labels[col_idx]+' - '+row_labels[row_idx])
                #ax.set_xlabel(row_labels[row_idx])
                #ax.set_xticklabels(qids_plot)
                if col_idx == 0:
                    ax.set_ylabel('AP (BM25)')
                ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
                ax.legend(loc='best', markerscale=0.5, fontsize=8)

        fig.suptitle(self.collection_name + ',qLen=%d' % query_length)
        output_fn = os.path.join(self.output_root, '%s-%d.%s' % (self.collection_name, query_length, oformat) )
        plt.savefig(output_fn, format=oformat, bbox_inches='tight', dpi=400)

    def get_rel_all_features(self, all_tfs, details_data):
        """
        all_tfs: {
            qid1: [[1, 3], [4, 6], [8, 5] ...],
            qid2: ...
            ...
        }
        """
        # for qid, tfs in all_tfs.items():
        #     print qid, tfs, details_data[qid][2], np.argmax(details_data[qid][2]), np.argmin(details_data[qid][2])
        #     print np.mean(tfs, axis=0)[np.argmax(details_data[qid][2])]
        #     print np.mean(tfs, axis=0)[np.argmin(details_data[qid][2])]
        #     print np.mean(tfs, axis=0)[np.argmin(details_data[qid][2])] - np.mean(tfs, axis=0)[np.argmax(details_data[qid][2])]
        #     print np.mean(tfs, axis=0)[np.argmin(details_data[qid][2])] / np.mean(tfs, axis=0)[np.argmax(details_data[qid][2])]
        #     print np.mean(tfs) 
        #     print np.count_nonzero(np.fabs(np.diff(tfs)) == 0)*1. / np.fabs(np.diff(tfs)).size,
        #     raw_input()
        all_labels = [
            'avg TF of small IDF term',
            'avg TF of large IDF term',
            'avg TF diff',
            'avg TF ratio',
            'avg of all terms',
            'inner doc TF diff == 0',
            'inner doc TF diff [1,5]',
            'inner doc TF diff [5,10]',
            'inner doc TF diff > 10',
        ]
        data = []
        for qid, tfs in all_tfs.items():
            if tfs.size == 0:
                continue
            col_means = np.mean(tfs, axis=0)
            row_diffs = np.diff(tfs)
            abs_row_diffs = np.fabs(row_diffs)
            dfs = details_data[qid][2]
            data.append([
                col_means[np.argmax(dfs)], # avg TF of terms with smaller IDF
                col_means[np.argmin(dfs)], # avg TF of terms with larger IDF
                col_means[np.argmin(dfs)] - col_means[np.argmax(dfs)], # diff
                np.mean(tfs, axis=0)[np.argmin(dfs)] / col_means[np.argmax(dfs)], # ratio
                np.mean(tfs), # all counts avg
                np.count_nonzero(abs_row_diffs == 0)*1. / abs_row_diffs.size,  
                np.count_nonzero([1 if v >= 1 and v <= 5 else 0 for v in abs_row_diffs])*1. / abs_row_diffs.size, 
                np.count_nonzero([1 if v > 5 and v <= 10 else 0 for v in abs_row_diffs])*1. / abs_row_diffs.size, 
                np.count_nonzero(abs_row_diffs > 10)*1. / abs_row_diffs.size, 
            ])
        return all_labels, np.array(data).transpose()

    def plot_only_rel_with_all_qterms(self, data, details_data, rel_data, query_length=2, oformat='png'):
        all_xlabels, all_xaxis = self.get_rel_all_features(data, details_data)
        qids = data.keys()
        yaxis = [float(rel_data[qid]['AP']['okapi'][1]) for qid in qids if qid in rel_data] # yaxis is the performance, e.g. AP
        num_cols = min(3, len(all_xlabels))
        num_rows = int(math.ceil(len(all_xlabels)*1.0/num_cols))
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, sharex=False, sharey=True, figsize=(3*num_cols, 3*num_rows))
        font = {'size' : 10}
        plt.rc('font', **font)
        row_idx = 0
        col_idx = 0
        for xaxis in all_xaxis:
            if num_rows > 1:
                ax = axs[row_idx][col_idx]
            else:
                if num_cols > 1:
                    ax = axs[col_idx]
                else:
                    ax = axs
            zipped = zip(qids, xaxis, yaxis)
            zipped.sort(key=itemgetter(1))
            qids_plot = np.array(zip(*zipped)[0])
            xaxis_plot = np.array(zip(*zipped)[1])
            yaxis_plot = np.array(zip(*zipped)[2])
            legend = 'pearsonr:%.4f' % (scipy.stats.pearsonr(xaxis_plot, yaxis_plot)[0])
            ax.plot(xaxis_plot, yaxis_plot, marker='o', ms=4, ls='None', label=legend)
            ax.set_title(all_xlabels[row_idx*num_cols+col_idx])
            #ax.set_xlabel(row_labels[row_idx])
            #ax.set_xticklabels(qids_plot)
            if col_idx == 0:
                ax.set_ylabel('AP (BM25)')
            ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
            ax.legend(loc='best', markerscale=0.5, fontsize=8)

            col_idx += 1
            if col_idx >= num_cols:
                row_idx += 1
                col_idx = 0

        fig.suptitle(self.collection_name + ',qLen=%d' % query_length)
        output_fn = os.path.join(self.output_root, '%s-%d-subrel.%s' % (self.collection_name, query_length, oformat) )
        plt.savefig(output_fn, format=oformat, bbox_inches='tight', dpi=400)


    def cal_map(self, rels_ranking_list, cutoff=1000, total_rel_cnt=0):
        """
        rels_ranking_list: a list only contains 0 and numbers larger than 0, 
        indicating the relevant info of the ranking list.
        """
        cur_rel = 0
        s = 0.0
        for i, ele in enumerate(rels_ranking_list):
            is_rel = int(ele) > 0
            if is_rel:
                cur_rel += 1
                s += cur_rel*1.0/(i+1)
            if i >= cutoff:
                break
        if total_rel_cnt == 0:
            return 0
        return s/total_rel_cnt

    def dir(self, terms, tfs, dfs, doclens, rels, mu=2500, which_term=0):
        if which_term > 0 and which_term < tfs.shape[0]:
            new_tfs = []
            for i in range(tfs.shape[0]):
                if i == which_term-1:
                    new_tfs.append(tfs[i])
                else:
                    new_tfs.append([0 if n > 0 else 0 for n in tfs[i]])
            tfs = np.array(new_tfs)
        cs = CollectionStats(self.collection_path)
        total_terms_cnt = cs.get_total_terms()
        terms_collection_occur = np.reshape(np.repeat([cs.get_term_collection_occur(t)*1./total_terms_cnt for t in terms], tfs.shape[1]), tfs.shape)
        r = np.log((tfs+mu*terms_collection_occur)/(doclens+mu))
        return np.sum(r, axis=0) if which_term==0 else r[which_term-1]

    def okapi(self, terms, tfs, dfs, doclens, rels, b=0.25, which_term=0):
        """
        which_term can determine which term is used to compute the 
        score. If it is 0 then all terms will be used, otherwise only 
        the selected term is used. 
        which_term indicates the row index of tfs.
        """
        cs = CollectionStats(self.collection_path)
        if which_term > 0 and which_term < tfs.shape[0]:
            new_tfs = []
            for i in range(tfs.shape[0]):
                if i == which_term-1:
                    new_tfs.append(tfs[i])
                else:
                    new_tfs.append([0 if n > 0 else 0 for n in tfs[i]])
            tfs = np.array(new_tfs)
        idfs = np.reshape(np.repeat(np.log((cs.get_doc_counts()-dfs+0.5)/(dfs+0.5)), tfs.shape[1]), tfs.shape)
        avdl = cs.get_avdl()
        k1 = 1.2
        r = (k1+1.0)*tfs/(tfs+k1*(1-b+b*doclens*1.0/avdl))*idfs
        return np.sum(r, axis=0)

    def plot_only_rel_tf_relationship(self, details_data, details_rel_data, 
            rel_data, query_length=2, plot_option=1, oformat='png'):
        rel_tf_stats = RelTFStats(self.collection_path)
        if query_length == 0:
            queries = Query(self.collection_path).get_queries()
        else:
            queries = Query(self.collection_path).get_queries_of_length(query_length)
        queries = {ele['num']:ele['title'] for ele in queries}
        rel_docs = Judgment(self.collection_path).get_relevant_docs_of_some_queries(queries.keys(), 1, 'dict')
        queries = {k:v for k,v in queries.items() if k in rel_docs and len(rel_docs[k]) > 0}

        rel_data = rel_tf_stats.get_data(queries.keys())
        cs = CollectionStats(self.collection_path)
        doc_details = GenDocDetails(self.collection_path)

        model_mapping = {
            'okapi': self.okapi,
            'dir': self.dir,
        }
        ranking_models = [('okapi', 'x'), ('dir', '^')]
        all_performances = {k:{'all': {}, 'higher-IDF': {}, 'lower-IDF': {}} for k in model_mapping}
        num_cols = min(4, len(details_rel_data)+1) # extra one for explanations
        num_rows = int(math.ceil((len(details_rel_data)+1)*1.0/num_cols))
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, sharex=False, sharey=False, figsize=(3*num_cols+6, 3*num_rows+6))
        plt.rc('font', size=8)
        plt.rc('text', usetex=False)
        row_idx = 0
        col_idx = 0
        for qid in sorted(queries):
            #print qid
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
            terms = queries[qid].split()
            dfs = np.array([cs.get_term_df(t) for t in terms])
            qid_details = {row['docid']:row for row in doc_details.get_qid_details(qid)}
            rel_tfs = details_rel_data[qid][1]
            #dfs = details_rel_data[qid][2]
            doclens = details_rel_data[qid][3]
            all_tfs = details_data[qid][1]
            all_dfs = details_data[qid][2]
            all_doclens = details_data[qid][3]
            all_rels = details_data[qid][4]
            if dfs.size == 0:
                continue
            idfs = np.log((cs.get_doc_counts() + 1)/(dfs+1e-4))
            smaller_idf_idx = np.argmax(dfs)
            larger_idf_idx = np.argmin(dfs)

            rel_xaxis = rel_tfs[smaller_idf_idx,:]
            rel_yaxis = rel_tfs[larger_idf_idx,:]
            rel_counts = collections.Counter(zip(rel_xaxis, rel_yaxis))
            all_xaxis = all_tfs[smaller_idf_idx,:]
            all_yaxis = all_tfs[larger_idf_idx,:]
            all_counts = collections.Counter(zip(all_xaxis, all_yaxis))
            prob_counts = {k:rel_counts[k]*1./v for k,v in all_counts.items() if k in rel_counts}
            nonrel_counts = {k:v for k,v in all_counts.items() if k not in rel_counts}
            if plot_option == 1:
                counts = rel_counts
            elif plot_option == 2:
                counts = prob_counts
            elif plot_option == 3:
                counts = all_counts
            xaxis_plot, yaxis_plot = zip(*counts.keys())
            sizes = np.array(counts.values())
            max_value = max(max(xaxis_plot), max(yaxis_plot))
            scatter = ax.scatter(xaxis_plot, yaxis_plot, c=sizes, edgecolors='none')
            cbar = fig.colorbar(scatter, ax=ax)
            #cbar.ax.set_ylabel('Counts')
            # plot model top ranked docs
            legend_handlers = {}
            for model in ranking_models:
                model_name = model[0]
                marker = model[1]
                model_optimal = Performances(self.collection_path).load_optimal_performance([model_name])[0]
                indri_model_para = 'method:%s,' % model_optimal[0] + model_optimal[2]
                runfile_fn = os.path.join(self.collection_path, 'split_results', 'title_'+qid+'-'+indri_model_para)
                with open(runfile_fn) as runf:
                    model_ranking_list = runf.readlines()
                model_topranked_tfs = np.array([[float(t.split('-')[1]) for t in qid_details[line.split()[2]]['tf'].split(',')] for line in model_ranking_list[:20]])
                # if model_topranked_tfs.shape[1] > query_length:
                #     model_topranked_tfs = np.delete(model_topranked_tfs, 0, 1)
                model_topranked_tfs = np.transpose(model_topranked_tfs)
                subquery_perfms = {}
                with open(os.path.join(self.collection_path, 'subqueries/collected_results', qid)) as subf:
                    csvr = csv.reader(subf)
                    for row in csvr:
                        subquery_id = row[0]
                        subquery_len = int(subquery_id.split('-')[0])
                        if subquery_len == 1 and model_name in row[2]:
                            subquery_perfms[row[1]] = float(row[3])
                qid_optimal = Performances(self.collection_path).gen_optimal_performances_queries([model_name], [qid])
                all_performances[model_name]['all'][qid] = float(qid_optimal[0][1])
                all_performances[model_name]['higher-IDF'][qid] = subquery_perfms[terms[larger_idf_idx]]
                all_performances[model_name]['lower-IDF'][qid] = subquery_perfms[terms[smaller_idf_idx]]
                this_plot, = ax.plot(model_topranked_tfs[smaller_idf_idx][:], \
                    model_topranked_tfs[larger_idf_idx][:], marker, \
                    alpha=0.3, label='%s:%.3f(%.3f)(%.3f)' % (model_name, \
                        float(qid_optimal[0][1]), subquery_perfms[terms[larger_idf_idx]], \
                        subquery_perfms[terms[smaller_idf_idx]]))
                legend_handlers[this_plot] = HandlerLine2D(numpoints=1)

            ax.plot([0, max_value], [0, max_value], ls="dotted")
            ax.set_title(qid+':'+queries[qid])
            ax.set_xlabel('%s:%.2f' % (terms[smaller_idf_idx], idfs[smaller_idf_idx]), labelpad=-2)
            ax.set_ylabel('%s:%.2f' % (terms[larger_idf_idx], idfs[larger_idf_idx]), labelpad=0)
            ax.set_xlim([0, max_value])
            ax.set_ylim([0, max_value])
            ax.grid(ls='dotted')
            #ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
            ax.legend(handler_map=legend_handlers, loc='best', fontsize=6, markerscale=0.6, handletextpad=-0.5, frameon=False, framealpha=0.6)

        if num_rows > 1:
                ax = axs[row_idx][col_idx]
        else:
            if num_cols > 1:
                ax = axs[col_idx]
            else:
                ax = axs
        explanations = 'title: query id and query\n xaxis: tf of lower IDF term in rel docs\nyaxis: tf of higher IDF term in rel docs\n'
        explanations += 'xlabel: lower IDF term and its IDF\nylabel: higher IDF term and its IDF\n'
        explanations += 'scatter dots: TFs of rel docs\nx-markers: TFs of top 20 ranked docs of BM25\n^-markers: TFs of top 20 ranked docs of LM\n'
        explanations += 'legend: AP(AP of using higher IDF term only)\n(AP of using lower IDF term only)\n'
        explanations += 'overall AP:\n'
        for model in all_performances:
            explanations += '%s:%.3f,%.3f,%.3f\n' % (model, np.mean(all_performances[model]['all'].values()), 
                np.mean(all_performances[model]['higher-IDF'].values()), np.mean(all_performances[model]['lower-IDF'].values()))
        ax.text(0.5, 0.5, explanations, fontsize=6, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

        if plot_option == 1:
            output_fn = os.path.join(self.output_root, '%s-%d-tf_relation.%s' % (self.collection_name, query_length, oformat) )
        elif plot_option == 2:
            output_fn = os.path.join(self.output_root, '%s-%d-tf_rel_prob.%s' % (self.collection_name, query_length, oformat) )
        elif plot_option == 3:
            output_fn = os.path.join(self.output_root, '%s-%d-tf_norel.%s' % (self.collection_name, query_length, oformat) )
        plt.savefig(output_fn, format=oformat, bbox_inches='tight', dpi=400)


    def plot_all(self, query_length=2, oformat='png'):
        query_length = int(query_length)
        details_data = self.read_docdetails_data(query_length)
        details_rel_data = self.read_docdetails_data(query_length, only_rel=True)
        rel_data = self.read_rel_data(query_length)
        #prepared_data, rel_contain_alls = self.prepare_rel_data(query_length, details_data, rel_data)
        
        ##### plot all kinds of docs
        #self.plot_all_kinds_of_docs(prepared_data, details_data, rel_data, query_length, oformat)
        ##### plot ONLY the docs that contain all query terms
        #self.plot_only_rel_with_all_qterms(rel_contain_alls, details_data, rel_data, query_length, oformat)
        ##### plot the relationship between terms only, no ranking function involved...
        self.plot_only_rel_tf_relationship(details_data, details_rel_data, rel_data, query_length, 1, oformat)
        self.plot_only_rel_tf_relationship(details_data, details_rel_data, rel_data, query_length, 2, oformat)
        self.plot_only_rel_tf_relationship(details_data, details_rel_data, rel_data, query_length, 3, oformat)
    

    @staticmethod
    def plot_tdc_violation_batch(collection_paths_n_names, query_length=0, _type=1, ofn_format='png'):
        """
        Plot the TDC (Term Discrimination Constraint) violation.
        We pick top ranked documents from the original query and the subquery
        and plot the term frequencies (or the BM25 score) to see whether they are 
        concentrated around the diagnal line or near the axis lines.
        """
        results_root = os.path.join('../all_results', 'tdc_violation')
        if not os.path.exists(results_root):
            os.makedirs(results_root)
        all_qids = []
        for collection_path, collection_name in collection_paths_n_names:
            q = Query(collection_path)
            if query_length == 0:
                queries = q.get_queries()
            else:
                queries = q.get_queries_of_length(query_length)
            queries = {ele['num']:ele['title'] for ele in queries}

            gt_optimal, diff_sorted_qid = SubqueriesLearning(collection_path, collection_name)\
                                                .load_gt_optimal(queries.keys())
            for ele in diff_sorted_qid:
                if ele[-1] != 0.0:
                    qid = ele[0]
                    tmp = [collection_path, collection_name, qid, queries[qid], _type, os.path.join(results_root, qid+'_'+str(_type)+'.'+ofn_format), ofn_format]
                    all_qids.append(tmp)
        return all_qids

    def get_runfiles_n_performances(self, req_qid, model='okapi'):
        subquery_learn_class = SubqueriesLearning(self.collection_path, self.collection_name)
        results = {}
        for fn in os.listdir(subquery_learn_class.subqueries_performance_root):
            fn_split = fn.split('_')
            qid = fn_split[0]
            if qid != req_qid:
                continue
            subquery_id = fn_split[1]
            model_para = fn_split[2]
            if model not in model_para:
                continue
            print fn
            try:
                with open(os.path.join(subquery_learn_class.subqueries_performance_root, fn)) as f:
                    first_line = f.readline()
                    ap = float(first_line.split()[-1])
            except:
                continue
            with open(os.path.join(subquery_learn_class.subqueries_runfiles_root, fn)) as f:
                first_100_lines = f.readlines()[:100]
            results[subquery_id] = {'ap': ap, 'first_lines': first_100_lines}
        return results

    def sort_subquery_id(self, subquery_id):
        return int(subquery_id.split('-')[0])+float(subquery_id.split('-')[1])/10.0

    def get_terms_scores_for_tdc_violation(self, ranking_list, _type=1):
        all_scores = []
        line_idx = 0
        for line in ranking_list:
            line = line.strip()
            if line:
                row = line.split()
                tf_details = row[1]
                terms = [ele.split('-')[0] for ele in tf_details.split(',')]
                tfs = [float(ele.split('-')[1]) for ele in tf_details.split(',')]
                dl = float(row[-1].split(',')[0].split(':')[1])
                if _type == 1: # simple TF
                    scores = tfs
                elif _type == 2: # BM25
                    scores = [tf*cs.get_term_logidf1(terms[i])*2.2/(tf+1.2*(1-model_para+model_para*dl/cs.get_avdl())) for i, tf in enumerate(tfs)]
                all_scores.append(scores)
            line_idx += 1
            if line_idx >= 100:
                break
        return all_scores

    def plot_tdc_violation(self, runfiles_n_performances, subquery_mapping, _type, output_fn, ofn_format='png'):
        num_cols = min(4, len(runfiles_n_performances)+1) # extra one for explanations
        num_rows = int(math.ceil((len(runfiles_n_performances)+1)*1.0/num_cols))
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, sharex=False, sharey=False, figsize=(3*num_cols+3, 3*num_rows+3))
        plt.rc('font', size=8)
        plt.rc('text', usetex=False)
        row_idx = 0
        col_idx = 0
        for subquery_id in sorted(runfiles_n_performances, key=self.sort_subquery_id):
            #print qid
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
            all_scores = self.get_terms_scores_for_tdc_violation(runfiles_n_performances[subquery_id]['first_lines'])
            all_scores = np.array(all_scores)
            if all_scores.shape[1] > 3:
                continue
            ax.plot(all_scores)
        plt.savefig(output_fn, format=ofn_format, bbox_inches='tight', dpi=400)

    def plot_tdc_violation_atom(self, qid, query, _type, output_fn, ofn_format='png'):
        q_class = Query(self.collection_path)
        subquery_learn_class = SubqueriesLearning(self.collection_path, self.collection_name)
        queries = {ele['num']:ele['title'] for ele in q_class.get_queries()}
        with open(os.path.join(subquery_learn_class.subqueries_mapping_root, qid)) as f:
            subquery_mapping = json.load(f)
        rel_docs = Judgment(self.collection_path).get_relevant_docs_of_some_queries([qid], format='dict')[qid]
        cs = CollectionStats(self.collection_path)
        rps = self.get_runfiles_n_performances(qid)
        self.plot_tdc_violation(rps, subquery_mapping, _type, output_fn, ofn_format)
