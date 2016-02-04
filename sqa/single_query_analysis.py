import sys,os
import math
import argparse
import json
import ast
import subprocess
import time
from subprocess import Popen, PIPE
from datetime import datetime
from operator import itemgetter
import multiprocessing
import re

import inspect
from inspect import currentframe, getframeinfo

import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.stats import entropy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils/'))
from query import Query
from results_file import ResultsFile
from judgment import Judgment
from evaluation import Evaluation
from utils import Utils
from collection_stats import CollectionStats
from baselines import Baselines
import indri
from MPI import MPI


def IndriRunQuery(query_file_path, output_path, method=None):
    """
    This function should be outside the class so that it can be executed by 
    children process.
    """
    frameinfo = getframeinfo(currentframe())
    print frameinfo.filename+':'+str(frameinfo.lineno),
    print query_file_path, method, output_path
    with open(output_path, 'wb') as f:
        if method:
            subprocess.Popen(['IndriRunQuery', query_file_path, method], bufsize=-1, stdout=f)
        else:
            subprocess.Popen(['IndriRunQuery', query_file_path], bufsize=-1, stdout=f)
        f.flush()
        os.fsync(f.fileno())
        time.sleep(3)


def process_json(c, r):
    json_results = {}
    c_tag = c[3:]
    #print c_tag
    cs = CollectionStats(c)
    doc_cnt = cs.get_doc_counts()
    single_queries = Query(c).get_queries_of_length(1)
    qids = [ele['num'] for ele in single_queries]
    #print qids
    judgment = Judgment(c).get_relevant_docs_of_some_queries(qids, 1, 'dict')
    r_tag = r
    #print r_tag
    results = ResultsFile(os.path.join(c, r)).get_results_of_some_queries(qids)
    #print qids, results.keys()
    for qid, qid_results in results.items():
        this_key = c+','+qid+','+r_tag
        json_results[this_key] = []
        non_rel_cnt = 0
        #print qid
        qid_doc_stats = cs.get_qid_doc_statistics(qid)
        maxTF = cs.get_term_maxTF(cs.get_idf(qid).split('-')[0])
        for idx, ele in enumerate(qid_results):
            docid = ele[0]
            score = ele[1]
            if docid in judgment[qid]:
                json_results[this_key].append(\
                    (docid, score, qid_doc_stats[docid]['TOTAL_TF'], \
                    maxTF, non_rel_cnt, non_rel_cnt*1./doc_cnt))
            else:
                non_rel_cnt += 1
    #print json_results
    return json_results

def get_external_docno(collection_path, docid):
    return CollectionStats(collection_path).get_external_docid(docid)


def callwrapper(func, args):
    return func(*args)

def pool_call(args):
    return callwrapper(*args)


class SingleQueryAnalysis(object):
    def __init__(self):
        self.all_results_root = '../../all_results' # assume working dir is this dir
        self.batch_root = '../../batch/'

    #####################################
    def pre_screen_ax(self, performances_comparisons):
        """
        pre-screen the queries: 
        1. first sort their best MAP performances.
        2. then group 4 to one figure.
        """
        r = []

        xaxis = np.arange(1, 101)
        for qid in performances_comparisons:
            yaxis = [performances_comparisons[qid][x] for x in xaxis]
            best_map = max(yaxis)
            best_map_x = yaxis.index(best_map)+1 # best map's x coordinate
            r.append([qid, yaxis, best_map, best_map_x])

        r.sort(key=itemgetter(2, 0))
        return r



    def plot_compare_tf3_performances(self, performances_comparisons, collection_path):
        baseline_best_results = Baselines(collection_path).get_baseline_best_results()
        #print baseline_best_results
        collection_name = collection_path.split('/')[-1]
        screened_queries = self.pre_screen_ax(performances_comparisons)
        rows_cnt = int( math.ceil( len(performances_comparisons)/4. ) )
        fig, axs = plt.subplots(nrows=rows_cnt, ncols=1, sharex=False, sharey=False, figsize=(6, 3.*rows_cnt))
        font = {'size' : 8}
        plt.rc('font', **font)
        xaxis = np.arange(1, 101)
        if rows_cnt > 1:
            ax = axs[0]
        else:
            ax = axs
        idx = 0
        ax_idx = 0
        cs = CollectionStats(collection_path)
        single_queries = Query(collection_path).get_queries_of_length(1)
        queries = {ele['num']:ele['title'] for ele in single_queries}

        for ele in screened_queries:
            qid = ele[0]
            best_map = ele[2]
            best_map_x = ele[3]
            maxTF = cs.get_term_maxTF(queries[qid])
            xaxis = np.arange(1, min(maxTF+1, 101) )
            yaxis = ele[1][:len(xaxis)]
            #print len(xaxis), len(yaxis)
            line, = ax.plot(xaxis, yaxis, label=qid)
            line_color = line.get_color()
            ax.vlines(best_map_x, 0, best_map, linestyles='dotted', linewidth=.5, colors=line_color)

            ax.hlines(baseline_best_results[qid], 0, maxTF+1, linestyles='-', linewidth=.5, colors=line_color)
            idx += 1
            if idx >= 4:
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                ax_idx += 1
                ax = axs[ax_idx]
                idx = 0
            if ax_idx == 0:
                ax.set_title("TF analysis for single term queries (%s)" % collection_name)
            ax.set_ylabel("Mean Average Precision")
            
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_xlabel("TF fade points: if TF<=fade point then Score=TF, if TF>fade point then Score=fade_point+log(TF)")
        plt.savefig( os.path.join(self.all_results_root, 'tf3_compare', collection_name+'-compare_tf3_performances.png'), 
            format='png', bbox_inches='tight', dpi=400)

    def compare_tf3_with_baselines(self, performances_comparisons, collection_path):
        output_path = 'tf3_compare'
        baseline_best_results = Baselines(collection_path).get_baseline_best_results()
        collection_name = collection_path.split('/')[-1]
        output_fn = os.path.join(self.all_results_root, 'tf3_compare', collection_name+'-compare_tf3_performances.csv')
        with open(output_fn, 'wb') as f:
            f.write('qid,tf3_MAP, best_baseline_MAP\n')
            for qid in performances_comparisons:
                f.write('%s,%f,%f\n' % (qid, performances_comparisons[qid][1], baseline_best_results[qid]) )



    def indri_run_query_atom(self, para_file):
        with open(para_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    row = line.split()
                    indri.IndriRunQuery(row[0], row[2], row[1])

    def run_tf3(self, collections_path=[]):
        frameinfo = getframeinfo(currentframe())
        current_function_name = inspect.stack()[0][3]
        for c in collections_path:
            collection_name = c.split('/')[-1] if c.split('/')[-1] else c.split('/')[-2]
            single_queries = Query(c).get_queries_of_length(1)
            qids = [ele['num'] for ele in single_queries]
            performances_comparisons = {}
            all_paras = []
            for i in xrange(1, 101):
                q_path = os.path.join(c, 'standard_queries')
                r_path = os.path.join(c, 'results', 'tf3_%d' % i)
                if not os.path.exists(r_path):
                    all_paras.append((q_path, '-rule=method:tf-3,f:%d'%i, r_path))
            if all_paras:
                #print all_paras
                MPI().gen_batch_framework(os.path.join(self.batch_root, collection_name, 'bin'), 
                    current_function_name, frameinfo.filename, '111', 
                    all_paras, 
                    os.path.join(self.batch_root, collection_name, 'misc', current_function_name), 
                    para_alreay_split=False,
                    add_node_to_para=False,
                    run_after_gen=True,
                    memory='4G'
                )
            else:
                print 'Nothing to RUN for '+c        

    def compare_tf3_performances(self, collections_path=[]):
        for c in collections_path:
            single_queries = Query(c).get_queries_of_length(1)
            qids = [ele['num'] for ele in single_queries]
            performances_comparisons = {}
            for i in xrange(1, 101):
                r_path = os.path.join(c, 'results', 'tf3_%d' % i)
                performances = Evaluation(c, r_path).get_all_performance_of_some_queries(qids=qids, return_all_metrics=False)
                for qid in performances:
                    if qid not in performances_comparisons:
                        performances_comparisons[qid] = {}
                    performances_comparisons[qid][i] = performances[qid]['map']
            # plot the per-query performance of TF3.
            # this may not be meaningful for single-term queries since 
            # the performances do not change for different fade points.
            # So for single-term queries, just compare it with the best results of baselines
            
            #self.plot_compare_tf3_performances(performances_comparisons, c)
            self.compare_tf3_with_baselines(performances_comparisons, c)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-111', '--indri_run_query_atom', nargs=1,
                       help='indri_run_query_atom')

    parser.add_argument('-1', '--plot_cost_of_rel_docs', action='store_true',
                       help='plot the cost of retrieving relevant documents')

    parser.add_argument('-2', '--batch_run_okapi_pivoted_without_idf', action='store_true',
                       help='batch run okapi pivoted without idf')

    parser.add_argument('-3', '--plot_single', nargs=3,
                       help='plot single figure')

    parser.add_argument('-4', '--plot_tfc_constraints', nargs='+',
                       help='plot the relevant document distribution. \
                       Take TF as an example and we start from the simplest case when \
                       |Q|=1.  We could leverage an existing collection \
                       and estimate P( c(t,D)=x | D is a relevant document), \
                       where x = 0,1,2,...maxTF(t). ')

    parser.add_argument('-51', '--run_tf3', nargs='+',
                       help='Batch run TF3 results')

    parser.add_argument('-52', '--compare_tf3_performances', nargs='+',
                       help='Compare all the performances of tf3 results. \
                       TF3: if tf<fade_point: score=tf; if tf>fade_point: score=log(tf).')

    args = parser.parse_args()

    if args.indri_run_query_atom:
        SingleQueryAnalysis().indri_run_query_atom(args.indri_run_query_atom[0])

    if args.plot_cost_of_rel_docs:
        SingleQueryAnalysis().plot_cost_of_rel_docs()

    if args.batch_run_okapi_pivoted_without_idf:
        SingleQueryAnalysis().batch_run_okapi_pivoted_without_idf()

    if args.plot_single:
        SingleQueryAnalysis().plot_single(args.plot_single[0], 
            args.plot_single[1], args.plot_single[2])

    if args.plot_tfc_constraints:
        SingleQueryAnalysis().plot_tfc_constraints(args.plot_tfc_constraints)

    if args.run_tf3:
        SingleQueryAnalysis().run_tf3(args.run_tf3)

    if args.compare_tf3_performances:
        SingleQueryAnalysis().compare_tf3_performances(args.compare_tf3_performances)
