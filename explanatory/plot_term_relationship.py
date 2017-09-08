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
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.weight'] = 'bold'
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from mpl_toolkits.mplot3d import Axes3D


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

    def plot_only_rel_tf_relationship_mul(self, details_data, details_rel_data, 
            rel_data, plot_option=1, method=1):
        if method == 1:
            method_name = 'TF'
        elif method == 2:
            method_name = 'BM25'
        queries = Query(self.collection_path).get_queries()
        queries = {ele['num']:ele['title'] for ele in queries}
        rel_tf_stats = RelTFStats(self.collection_path)
        rel_data = rel_tf_stats.get_data(queries.keys())
        cs = CollectionStats(self.collection_path)
        doc_details = GenDocDetails(self.collection_path)
        for qid in sorted(queries):
            try:
                terms = queries[qid].split()
                dfs = np.array([cs.get_term_df(t) for t in terms])          
                qid_details = {row['docid']:row for row in doc_details.get_qid_details(qid)}
            
                rel_tfs = details_rel_data[qid][1]
                #dfs = details_rel_data[qid][2]
                doclens = details_rel_data[qid][3]
                all_tfs = details_data[qid][1]
                if method == 2: # BM25
                    okapi_optimal = Performances(self.collection_path).load_optimal_performance(['okapi'])[0]
                    okapi_para = 'method:%s,' % okapi_optimal[0] + okapi_optimal[2]
                    optimal_b = float(okapi_optimal[2].split(':')[1])
                    tf_col_idx = 0
                    tmp_all_tfs = []
                    tmp_rel_tfs = []
                    for tf_col in all_tfs:
                        tf_col = tf_col*cs.get_term_logidf1(terms[tf_col_idx])*2.2/(tf_col+1.2*(1-optimal_b+optimal_b*doclens[tf_col_idx]/cs.get_avdl()))
                        rel_tf_col = rel_tfs[tf_col_idx]*cs.get_term_logidf1(terms[tf_col_idx])*2.2/(rel_tfs[tf_col_idx]+1.2*(1-optimal_b+optimal_b*doclens[tf_col_idx]/cs.get_avdl()))
                        tmp_all_tfs.append(tf_col)
                        tmp_rel_tfs.append(rel_tf_col)
                        tf_col_idx += 1
                    all_tfs = np.array(tmp_all_tfs)
                    rel_tfs = np.array(tmp_rel_tfs)
                
                all_dfs = details_data[qid][2]
                all_doclens = details_data[qid][3]
                all_rels = details_data[qid][4]
                if dfs.size == 0:
                    continue
                idfs = np.log((cs.get_doc_counts() + 1)/(dfs+1e-4))
                output_root = os.path.join(self.output_root, self.collection_name)
                if not os.path.exists(output_root):
                    os.makedirs(output_root)
                output_fn = os.path.join(output_root, '%s-%s-%s.json' % (self.collection_name, qid, method_name))
                d = {
                    'terms': terms,
                    'idfs': idfs.tolist(),
                    'rel_tfs': rel_tfs.transpose().tolist(),
                    'all_tfs': all_tfs.transpose().tolist()
                }
                with open(output_fn, 'wb') as f:
                    json.dump(d, f, indent=2)
            except:
                print 'We have some problems with qid: %s' % qid

    def plot_only_rel_tf_relationship(self, details_data, details_rel_data, 
            rel_data, query_length=2, plot_option=1, method=1, oformat='png'):
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
            if method == 2: # BM25
                okapi_optimal = Performances(self.collection_path).load_optimal_performance(['okapi'])[0]
                okapi_para = 'method:%s,' % okapi_optimal[0] + okapi_optimal[2]
                optimal_b = float(okapi_optimal[2].split(':')[1])
                tf_col_idx = 0
                tmp_all_tfs = []
                tmp_rel_tfs = []
                print terms, doclens
                for tf_col in all_tfs:
                    tf_col = tf_col*cs.get_term_logidf1(terms[tf_col_idx])*2.2/(tf_col+1.2*(1-optimal_b+optimal_b*doclens[tf_col_idx]/cs.get_avdl()))
                    rel_tf_col = rel_tfs[tf_col_idx]*cs.get_term_logidf1(terms[tf_col_idx])*2.2/(rel_tfs[tf_col_idx]+1.2*(1-optimal_b+optimal_b*doclens[tf_col_idx]/cs.get_avdl()))
                    tmp_all_tfs.append(tf_col)
                    tmp_rel_tfs.append(rel_tf_col)
                    tf_col_idx += 1
                all_tfs = np.array(tmp_all_tfs)
                rel_tfs = np.array(tmp_rel_tfs)
            
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
            ranking_models = [('okapi', 'x'), ('dir', '^')]
            for model in ranking_models:
                model_name = model[0]
                if method == 2 and model_name != 'okapi':
                    continue
                marker = model[1]
                model_optimal = Performances(self.collection_path).load_optimal_performance([model_name])[0]
                indri_model_para = 'method:%s,' % model_optimal[0] + model_optimal[2]
                runfile_fn = os.path.join(self.collection_path, 'split_results', 'title_'+qid+'-'+indri_model_para)
                with open(runfile_fn) as runf:
                    model_ranking_list = runf.readlines()
                model_topranked_tfs = np.array([[float(t.split('-')[1]) for t in qid_details[line.split()[2]]['tf'].split(',')] for line in model_ranking_list[:50]])
                model_topranked_tfs = np.transpose(model_topranked_tfs)
                if method != 1:
                    optimal_para = float(model_optimal[2].split(':')[1])
                    tf_col_idx = 0
                    tmp_model_tfs = []
                    for tf_col in model_topranked_tfs:
                        if model_name == 'okapi':
                            tf_col = tf_col*cs.get_term_logidf1(terms[tf_col_idx])*2.2/(tf_col+1.2*(1-optimal_para+optimal_para*doclens[tf_col_idx]/cs.get_avdl()))
                        elif model_name == 'dir':
                            tf_col = np.log((tf_col+optimal_para*cs.get_term_collection_occur(terms[tf_col_idx])/cs.get_total_terms())/(optimal_para+doclens[tf_col_idx]))
                        tmp_model_tfs.append(tf_col)
                        tf_col_idx += 1
                    model_topranked_tfs = np.array(tmp_model_tfs)
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

        if method == 1:
            method_name = 'TF'
        elif method == 2:
            method_name = 'BM25'
        if plot_option == 1:
            output_fn = os.path.join(self.output_root, '%s-%d-tf_relation-%s.%s' % (self.collection_name, query_length, method_name, oformat) )
        elif plot_option == 2:
            output_fn = os.path.join(self.output_root, '%s-%d-tf_rel_prob-%s.%s' % (self.collection_name, query_length, method_name, oformat) )
        elif plot_option == 3:
            output_fn = os.path.join(self.output_root, '%s-%d-tf_norel-%s.%s' % (self.collection_name, query_length, method_name, oformat) )
        plt.savefig(output_fn, format=oformat, bbox_inches='tight', dpi=400)


    def plot_rel_tf_relationship_qids(self, details_data, details_rel_data, 
            rel_data, query_length=2, method=1, oformat='png'):
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
        all_performances = {k:{'all': {}, 'higher-IDF': {}, 'lower-IDF': {}} for k in model_mapping}

        col_factor = 1
        json_output = {}
        for qid in sorted(queries):
            fig, axs = plt.subplots(nrows=1, ncols=col_factor, sharex=False, sharey=False, figsize=(3*col_factor+2, 3*1+1))
            plt.rc('font', size=15)
            plt.rc('text', usetex=False)
            terms = queries[qid].split()
            dfs = np.array([cs.get_term_df(t) for t in terms])
            qid_details = {row['docid']:row for row in doc_details.get_qid_details(qid)}
            rel_tfs = details_rel_data[qid][1]
            #dfs = details_rel_data[qid][2]
            doclens = details_rel_data[qid][3]
            all_tfs = details_data[qid][1]
            if method == 2: # BM25
                okapi_optimal = Performances(self.collection_path).load_optimal_performance(['okapi'])[0]
                okapi_para = 'method:%s,' % okapi_optimal[0] + okapi_optimal[2]
                optimal_b = float(okapi_optimal[2].split(':')[1])
                tf_col_idx = 0
                tmp_all_tfs = []
                tmp_rel_tfs = []
                for tf_col in all_tfs:
                    tf_col = tf_col*cs.get_term_logidf1(terms[tf_col_idx])*2.2/(tf_col+1.2*(1-optimal_b+optimal_b*doclens[tf_col_idx]/cs.get_avdl()))
                    rel_tf_col = rel_tfs[tf_col_idx]*cs.get_term_logidf1(terms[tf_col_idx])*2.2/(rel_tfs[tf_col_idx]+1.2*(1-optimal_b+optimal_b*doclens[tf_col_idx]/cs.get_avdl()))
                    tmp_all_tfs.append(tf_col)
                    tmp_rel_tfs.append(rel_tf_col)
                    tf_col_idx += 1
                all_tfs = np.array(tmp_all_tfs)
                rel_tfs = np.array(tmp_rel_tfs)
            
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
            if qid == '325' or qid == '394':
                json_output[qid] = {}
            for plot_option in range(3):
                if plot_option != 1:
                    continue
                if col_factor == 1:
                    ax = axs
                else:
                    ax = axs[plot_option]
                if plot_option == 0:
                    counts = rel_counts
                elif plot_option == 1:
                    counts = prob_counts
                elif plot_option == 2:
                    counts = all_counts
                xaxis_plot, yaxis_plot = zip(*counts.keys())
                sizes = np.array(counts.values())
                max_value = max(max(xaxis_plot), max(yaxis_plot))
                if qid == '325' or qid == '394':
                    json_output[qid][plot_option] = {
                        'x': list(xaxis_plot), 
                        'y': list(yaxis_plot),
                        'size': sizes.tolist(),
                        'max': max_value,
                        'query': queries[qid],
                        'xlabel': '%s:%.2f' % (terms[smaller_idf_idx], idfs[smaller_idf_idx]),
                        'ylabel': '%s:%.2f' % (terms[larger_idf_idx], idfs[larger_idf_idx])
                    }
                scatter = ax.scatter(xaxis_plot, yaxis_plot, c=sizes, edgecolors='none')
                cbar = fig.colorbar(scatter, ax=ax)
                #cbar.ax.set_ylabel('Counts')
                # plot model top ranked docs
                legend_handlers = {}
                ranking_models = [('okapi', 'x'), ('dir', '^')]
                for model in ranking_models:
                    continue
                    model_name = model[0]
                    if method == 2 and model_name != 'okapi':
                        continue
                    marker = model[1]
                    model_optimal = Performances(self.collection_path).load_optimal_performance([model_name])[0]
                    indri_model_para = 'method:%s,' % model_optimal[0] + model_optimal[2]
                    runfile_fn = os.path.join(self.collection_path, 'split_results', 'title_'+qid+'-'+indri_model_para)
                    with open(runfile_fn) as runf:
                        model_ranking_list = runf.readlines()
                    try:
                        model_topranked_tfs = np.array([[float(t.split('-')[1]) for t in qid_details[line.split()[2]]['tf'].split(',')] for line in model_ranking_list[:50]])
                    except:
                        continue
                    model_topranked_tfs = np.transpose(model_topranked_tfs)
                    if method != 1:
                        optimal_para = float(model_optimal[2].split(':')[1])
                        tf_col_idx = 0
                        tmp_model_tfs = []
                        for tf_col in model_topranked_tfs:
                            if model_name == 'okapi':
                                tf_col = tf_col*cs.get_term_logidf1(terms[tf_col_idx])*2.2/(tf_col+1.2*(1-optimal_para+optimal_para*doclens[tf_col_idx]/cs.get_avdl()))
                            elif model_name == 'dir':
                                tf_col = np.log((tf_col+optimal_para*cs.get_term_collection_occur(terms[tf_col_idx])/cs.get_total_terms())/(optimal_para+doclens[tf_col_idx]))
                            tmp_model_tfs.append(tf_col)
                            tf_col_idx += 1
                        model_topranked_tfs = np.array(tmp_model_tfs)
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

            if method == 1:
                method_name = 'TF'
            elif method == 2:
                method_name = 'BM25'
            output_root = os.path.join(self.output_root, self.collection_name)
            if not os.path.exists(output_root):
                os.makedirs(output_root)
            output_fn = os.path.join(output_root, '%s-%s.%s' % (qid, method_name, oformat) )
            plt.savefig(output_fn, format=oformat, bbox_inches='tight', dpi=400)
            plt.close()

            jsondata_output_root = os.path.join(self.output_root+'/json/')
            if not os.path.exists(jsondata_output_root):
                os.makedirs(jsondata_output_root)
            with open(os.path.join(jsondata_output_root, self.collection_name+'.json'), 'w') as f:
                json.dump(json_output, f, indent=2)

    def plot_all(self, query_length=2, method=1, oformat='png'):
        query_length = int(query_length)
        method = int(method)
        details_data = self.read_docdetails_data(query_length)
        details_rel_data = self.read_docdetails_data(query_length, only_rel=True)
        rel_data = self.read_rel_data(query_length)
        #prepared_data, rel_contain_alls = self.prepare_rel_data(query_length, details_data, rel_data)
        
        if query_length != 2:
            return self.plot_only_rel_tf_relationship_mul(details_data, details_rel_data, 
                rel_data, plot_option, method)
        else:
            # self.plot_only_rel_tf_relationship(details_data, details_rel_data, rel_data, query_length, 1, method, oformat)
            # self.plot_only_rel_tf_relationship(details_data, details_rel_data, rel_data, query_length, 2, method, oformat)
            # self.plot_only_rel_tf_relationship(details_data, details_rel_data, rel_data, query_length, 3, method, oformat)

            self.plot_rel_tf_relationship_qids(details_data, details_rel_data, rel_data, query_length, method, oformat)
    

    @staticmethod
    def plot_tdc_violation_batch(collection_paths_n_names, query_length=0, 
            top_n_docs=100, _type=1, terms_type=0, ofn_format='png'):
        """
        Plot the TDC (Term Discrimination Constraint) violation.
        We pick top ranked documents from the original query and the subquery
        and plot the term frequencies (or the BM25 score) to see whether they are 
        concentrated around the diagnal line or near the axis lines.
        """
        results_root = os.path.join('../../all_results', 'tdc_violation')
        if not os.path.exists(results_root):
            os.makedirs(results_root)
        all_qids = []
        if _type == 1:
            type_str = 'TF'
        elif _type == 2:
            type_str = 'BM25'
        elif _type == 3:
            type_str = 'LM'
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
                    if terms_type == 0:
                        output_fn = os.path.join(results_root, collection_name+'_'+qid+'_top'+str(top_n_docs)+'_'+type_str+'.'+ofn_format)
                    elif terms_type == 1:
                        output_fn = os.path.join(results_root, collection_name+'_'+qid+'_top'+str(top_n_docs)+'_'+type_str+'.json')
                    tmp = [collection_path, collection_name, qid, queries[qid], 
                        top_n_docs, _type, 
                        output_fn,
                        terms_type,
                        ofn_format]
                    all_qids.append(tmp)
        return all_qids

    def get_runfiles_n_performances(self, req_qid, terms_type=0, model='okapi'):
        subquery_learn_class = SubqueriesLearning(self.collection_path, self.collection_name)
        results = {}
        if terms_type == 0:
            runfiles_root = os.path.join(self.collection_path, 'subqueries', 'runfiles') 
        elif terms_type == 1:
            runfiles_root = os.path.join(self.collection_path, 'subqueries', 'runfiles_allterms')
        for fn in os.listdir(subquery_learn_class.subqueries_performance_root):
            fn_split = fn.split('_')
            qid = fn_split[0]
            if qid != req_qid:
                continue
            subquery_id = fn_split[1]
            model_para = fn_split[2]
            if model not in model_para:
                continue
            try:
                with open(os.path.join(subquery_learn_class.subqueries_performance_root, fn)) as f:
                    first_line = f.readline()
                    ap = float(first_line.split()[-1])
            except:
                continue
            with open(os.path.join(runfiles_root, fn)) as f:
                first_100_lines = f.readlines()[:100]
            results[subquery_id] = {'ap': ap, 'first_lines': first_100_lines}
        return results

    def sort_subquery_id(self, subquery_id):
        return int(subquery_id.split('-')[0])+float(subquery_id.split('-')[1])/10.0

    def get_terms_scores_for_tdc_violation(self, ranking_list, rel_docs, top_n_docs=20, _type=1):
        cs = CollectionStats(self.collection_path)
        methods = ['okapi', 'dir']
        optimal_performances = Performances(self.collection_path).load_optimal_performance(methods)
        indri_model_paras = []
        model_paras = []
        for ele in optimal_performances:
            indri_model_paras.append('method:%s,' % ele[0] + ele[2])
            model_paras.append(float(ele[2].split(':')[1]))
        all_scores = {'rel': [], 'nonrel': []}
        line_idx = 0
        for line in ranking_list:
            line = line.strip()
            if line:
                row = line.split()
                tf_details = row[1]
                docid = row[2]
                terms = [ele.split('-')[0] for ele in tf_details.split(',')]
                tfs = [float(ele.split('-')[1]) for ele in tf_details.split(',')]
                dl = float(row[-1].split(',')[0].split(':')[1])
                if _type == 1: # simple TF
                    scores = tfs
                elif _type == 2: # BM25
                    scores = [tf*cs.get_term_logidf1(terms[i])*2.2/(tf+1.2*(1-model_paras[0]+model_paras[0]*dl/cs.get_avdl())) for i, tf in enumerate(tfs)]
                all_scores['rel' if docid in rel_docs else 'nonrel'].append(scores)
            line_idx += 1
            if line_idx >= top_n_docs:
                break
        terms_n_idfs = [(t, cs.get_term_logidf1(t)) for t in terms]
        return terms_n_idfs, all_scores

    def output_tdc_data_for_all_terms(self, runfiles_n_performances, rel_docs, 
            subquery_mapping, top_n_docs, _type, output_fn):
        if len(subquery_mapping) != 7: # qlen == 3
            return
        all_data = {}
        for subquery_id in sorted(runfiles_n_performances, key=self.sort_subquery_id):
            terms_n_idfs, all_scores = self.get_terms_scores_for_tdc_violation(
                runfiles_n_performances[subquery_id]['first_lines'],
                rel_docs, 
                top_n_docs, 
                _type
            )
            all_scores['query'] = subquery_mapping[subquery_id]
            all_scores['ap'] = '%.4f' % runfiles_n_performances[subquery_id]['ap']
            all_data[subquery_id] = all_scores
        with open(output_fn, 'wb') as f:
            json.dump(all_data, f, indent=2)

    def plot_tdc_violation(self, runfiles_n_performances, rel_docs, 
            subquery_mapping, top_n_docs, _type, output_fn, terms_type=0, 
            ofn_format='png'):
        if len(subquery_mapping) > 7: # we can not draw the plots for query len > 3
            return
        
        only_plot_optimal_and_original = True
        if only_plot_optimal_and_original:
            allowed_subquery_id = []
            all_subquery_ids = [(k, v['ap']) for k,v in runfiles_n_performances.items()]
            all_subquery_ids.sort(key=itemgetter(1), reverse=True)
            query_len = 3 if len(runfiles_n_performances) == 7 else 2
            if all_subquery_ids[0][0] == str(query_len)+'-0':
                allowed_subquery_id.append(all_subquery_ids[1][0])
            else:
                allowed_subquery_id.append(all_subquery_ids[0][0])
            allowed_subquery_id.append(str(query_len)+'-0')
            not_allowed = [ele[0] for ele in all_subquery_ids if ele[0] not in allowed_subquery_id]
            for subquery_id in not_allowed:
                del(runfiles_n_performances[subquery_id])
            
            fig = plt.figure(figsize=(6, 3))
        else:
            if len(subquery_mapping) == 7:
                fig = plt.figure(figsize=(15, 9))
            elif len(subquery_mapping) == 3:
                fig = plt.figure(figsize=(12, 5))
        # num_cols = min(4, len(runfiles_n_performances)+1) # extra one for explanations
        # num_rows = int(math.ceil((len(runfiles_n_performances)+1)*1.0/num_cols))
        # fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, sharex=False, sharey=False, figsize=(3*num_cols+3, 3*num_rows+3))
        plt.rc('font', size=8)
        plt.rc('text', usetex=False)
        # row_idx = 0
        # col_idx = 0
        idx = 1
        all_all_scores = {}
        for subquery_id in sorted(runfiles_n_performances, key=self.sort_subquery_id):
            terms_n_idfs, all_scores = self.get_terms_scores_for_tdc_violation(
                runfiles_n_performances[subquery_id]['first_lines'],
                rel_docs, 
                top_n_docs, 
                _type
            )
            if len(terms_n_idfs) < 2:
                continue
            for k in all_scores:
                all_scores[k] = np.array(all_scores[k]).T
            if all_scores['nonrel'].shape[0] > 3:
                continue
            if all_scores['nonrel'].shape[0] == 1:
                if len(subquery_mapping) == 7:
                    if only_plot_optimal_and_original:
                        ax = fig.add_subplot(1, 2, 1)
                    else:
                        ax = fig.add_subplot(2, 4, idx)
                elif len(subquery_mapping) == 3:
                    ax = fig.add_subplot(1, 3, idx)
                if all_scores['rel'].shape[0] > 0:
                    ax.plot(all_scores['rel'][0], all_scores['rel'][0], 'go', alpha=0.5)
                ax.plot(all_scores['nonrel'][0], all_scores['nonrel'][0], 'ro', alpha=0.5)
            elif all_scores['nonrel'].shape[0] == 2:
                if len(subquery_mapping) == 7:
                    if only_plot_optimal_and_original:
                        ax = fig.add_subplot(1, 2, 1)
                    else:
                        ax = fig.add_subplot(2, 4, idx)
                elif len(subquery_mapping) == 3:
                    ax = fig.add_subplot(1, 3, idx)
                if all_scores['rel'].shape[0] > 0:
                    ax.plot(all_scores['rel'][0], all_scores['rel'][1], 'go', alpha=0.5)
                ax.plot(all_scores['nonrel'][0], all_scores['nonrel'][1], 'ro', alpha=0.5)
            elif all_scores['nonrel'].shape[0] == 3:
                if only_plot_optimal_and_original:
                    ax = fig.add_subplot(1, 2, 2, projection='3d')
                else:
                    ax = fig.add_subplot(2, 4, idx, projection='3d')
                if all_scores['rel'].shape[0] > 0:
                    ax.scatter(all_scores['rel'][0], all_scores['rel'][1], all_scores['rel'][2], c='g')
                ax.scatter(all_scores['nonrel'][0], all_scores['nonrel'][1], all_scores['nonrel'][2], c='r')
            else:
                continue
            max_value = max(np.amax(all_scores['rel']) if all_scores['rel'].shape[0] > 0 else 0, np.amax(all_scores['nonrel']))
            ax.set_title(subquery_mapping[subquery_id] + '(%.4f)' % runfiles_n_performances[subquery_id]['ap'])
            ax.set_xlabel('%s(%.2f)' % (terms_n_idfs[0][0], terms_n_idfs[0][1]), labelpad=0)
            ax.set_ylabel('%s(%.2f)' % (terms_n_idfs[1][0], terms_n_idfs[1][1]), labelpad=0)
            if all_scores['nonrel'].shape[0] == 3:
                ax.set_zlabel('%s(%.2f)' % (terms_n_idfs[2][0], terms_n_idfs[2][1]), labelpad=0)
            ax.set_xlim([0, max_value])
            ax.set_ylim([0, max_value])
            ax.grid(ls='dotted')
            idx += 1
            all_all_scores[subquery_id] = {'data': {},
                'max': max_value, 
                'title': subquery_mapping[subquery_id] + '(%.4f)' % runfiles_n_performances[subquery_id]['ap'], 
                'xlabel': '%s(%.2f)' % (terms_n_idfs[0][0], terms_n_idfs[0][1]),
                'ylabel': '%s(%.2f)' % (terms_n_idfs[1][0], terms_n_idfs[1][1]),
                'zlabel': '%s(%.2f)' % (terms_n_idfs[2][0], terms_n_idfs[2][1]) if all_scores['nonrel'].shape[0] == 3 else '',
            }
            for k, v in all_scores.items():
                all_all_scores[subquery_id]['data'][k] = v.tolist()
        with open(output_fn+'.json', 'w') as f:
            json.dump(all_all_scores, f, indent=2)
        plt.savefig(output_fn, format=ofn_format, bbox_inches='tight', dpi=400)

    def plot_tdc_violation_atom(self, qid, query, top_n_docs, _type, output_fn, terms_type=0, ofn_format='png'):
        """
        terms_type: 0-only terms in this subquery; 1-all terms in the original query
        """
        q_class = Query(self.collection_path)
        subquery_learn_class = SubqueriesLearning(self.collection_path, self.collection_name)
        queries = {ele['num']:ele['title'] for ele in q_class.get_queries()}
        with open(os.path.join(subquery_learn_class.subqueries_mapping_root, qid)) as f:
            subquery_mapping = json.load(f)
        rel_docs = Judgment(self.collection_path).get_relevant_docs_of_some_queries([qid], format='dict')[qid]
        rps = self.get_runfiles_n_performances(qid, terms_type)
        if terms_type == 0:
            self.plot_tdc_violation(rps, rel_docs, subquery_mapping, top_n_docs, _type, output_fn, ofn_format)
        elif terms_type == 1:
            self.output_tdc_data_for_all_terms(rps, rel_docs, subquery_mapping, top_n_docs, _type, output_fn)

