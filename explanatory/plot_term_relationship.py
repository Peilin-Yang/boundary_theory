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
        for qid, tfs in all_tfs.items():
            print qid, tfs, details_data[qid][2], np.argmax(details_data[qid][2]), np.argmin(details_data[qid][2])
            print np.mean(tfs, axis=0)[np.argmax(details_data[qid][2])]
            print np.mean(tfs, axis=0)[np.argmin(details_data[qid][2])]
            print np.mean(tfs, axis=0)[np.argmin(details_data[qid][2])] - np.mean(tfs, axis=0)[np.argmax(details_data[qid][2])]
            print np.mean(tfs, axis=0)[np.argmin(details_data[qid][2])] / np.mean(tfs, axis=0)[np.argmax(details_data[qid][2])]
            print np.mean(tfs) 
            print np.count_nonzero(np.fabs(np.diff(tfs)) == 0)*1. / np.fabs(np.diff(tfs)).size,
            raw_input()
        all_labels = [
            'avg TF of small IDF term',
            'avg TF of large IDF term',
            'avg TF diff',
            'avg TF ratio',
            'avg of all terms',
            'inner doc TF diff == 0',
            'inner doc TF diff <= 1',
            'inner doc TF diff <= 3',
            'inner doc TF diff <= 5',
            'inner doc TF diff <= 10',
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
                np.count_nonzero(abs_row_diffs == 0) / abs_row_diffs.size,  
                np.count_nonzero(abs_row_diffs <= 1) / abs_row_diffs.size, 
                np.count_nonzero(abs_row_diffs <= 3) / abs_row_diffs.size, 
                np.count_nonzero(abs_row_diffs <= 5) / abs_row_diffs.size, 
                np.count_nonzero(abs_row_diffs <= 10) / abs_row_diffs.size, 
                np.count_nonzero(abs_row_diffs > 10) / abs_row_diffs.size, 
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


    def plot_all(self, query_length=2, oformat='png'):
        query_length = int(query_length)
        details_data = self.read_docdetails_data(query_length)
        rel_data = self.read_rel_data(query_length)
        prepared_data, rel_contain_alls = self.prepare_rel_data(query_length, details_data, rel_data)
        
        ##### plot all kinds of docs
        #self.plot_all_kinds_of_docs(prepared_data, details_data, rel_data, query_length, oformat)
        ##### plot ONLY the docs that contain all query terms
        self.plot_only_rel_with_all_qterms(rel_contain_alls, details_data, rel_data, query_length, oformat)
