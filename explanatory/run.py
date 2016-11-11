#!/home/1471/ENV/bin/python
import sys,os
import math
import argparse
import json
import ast
import subprocess
import time
from subprocess import Popen, PIPE
import shlex
from datetime import datetime
from operator import itemgetter
import multiprocessing
import re
import csv
import shutil
import copy

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
from gen_doc_details import GenDocDetails
from plot_prob_rel import PlotRelProb
from plot_synthetic_map import PlotSyntheticMAP
from prints import Prints
import g
import ArrayJob

collection_root = '../../../reproduce/collections/'

def gen_batch_framework(para_label, batch_pythonscript_para, all_paras, \
        quote_command=False, memory='2G', max_task_per_node=50000, num_task_per_node=20):

    para_dir = os.path.join('batch_paras', '%s') % para_label
    if os.path.exists(para_dir):
        shutil.rmtree(para_dir)
    os.makedirs(para_dir)

    batch_script_root = 'bin'
    if not os.path.exists(batch_script_root):
        os.makedirs(batch_script_root)

    if len(all_paras) == 0:
        print 'Nothing to run for ' + para_label
        return

    tasks_cnt_per_node = min(num_task_per_node, max_task_per_node) if len(all_paras) > num_task_per_node else 1
    all_paras = [all_paras[t: t+tasks_cnt_per_node] for t in range(0, len(all_paras), tasks_cnt_per_node)]
    batch_script_fn = os.path.join(batch_script_root, '%s-0.qs' % (para_label) )
    batch_para_fn = os.path.join(para_dir, 'para_file_0')
    with open(batch_para_fn, 'wb') as bf:
        for i, ele in enumerate(all_paras):
            para_file_fn = os.path.join(para_dir, 'para_file_%d' % (i+1))
            bf.write('%s\n' % (para_file_fn))
            with open(para_file_fn, 'wb') as f:
                writer = csv.writer(f)
                writer.writerows(ele)
    command = 'python %s -%s' % (
        inspect.getfile(inspect.currentframe()), \
        batch_pythonscript_para
    )
    arrayjob_script = ArrayJob.ArrayJob()
    arrayjob_script.output_batch_qs_file(batch_script_fn, command, quote_command, True, batch_para_fn, len(all_paras), _memory=memory)
    run_batch_gen_query_command = 'qsub %s' % batch_script_fn
    subprocess.call( shlex.split(run_batch_gen_query_command) )


def gen_doc_details_batch():
    all_paras = []
    for q in g.query:
        collection_name = q['collection']
        collection_path = os.path.join(collection_root, collection_name)
        all_paras.extend(GenDocDetails(collection_path).batch_gen_doc_details_paras())

    #print all_paras
    gen_batch_framework('gen_doc_details', 'g2', all_paras)

def gen_doc_details(para_file):
    with open(para_file) as f:
        reader = csv.reader(f)
        for row in reader:
            collection_path = row[0]
            qid = row[1]
            query = row[2]
            GenDocDetails(collection_path).output_doc_details(qid, query)

def plot_tf_rel_all(paras):
    data = {}
    for q in g.query:
        collection_name = q['collection']
        collection_path = os.path.join(collection_root, collection_name)
        print 'Getting the data and plotting for %s' % collection_name
        results = PlotRelProb(collection_path, q['collection_formal_name']).wrapper(*paras)
        this_data = results[1]
        for x in this_data:
          if x not in data:
            data[x] = [0, 0]
          data[x][0] += this_data[x][0]
          data[x][1] += this_data[x][1]

    PlotRelProb(collection_path, q['collection_formal_name']).plot_with_data(data, 
      'All Collections(query len=%s)' % paras[0], '', *paras)    

def gen_plot_tf_rel_batch(paras):
    all_paras = []
    for q in g.query:
        collection_name = q['collection_formal_name']
        collection_path = os.path.join(collection_root, q['collection'])
        p = copy.deepcopy(paras)
        p.insert(0, collection_name)
        p.insert(0, collection_path)
        all_paras.append((p))
    gen_batch_framework('plot_rel_prob', '122', all_paras)

def plot_tf_rel_atom(para_file):
    with open(para_file) as f:
        reader = csv.reader(f)
        for row in reader:
            collection_path = row[0]
            collection_name = row[1]
            print row
            PlotRelProb(collection_path, collection_name).wrapper(*row[2:])

def gen_lambdarank_batch():
    all_paras = []
    
    with open('lambdarank.json') as f:
        methods = json.load(f)['methods']
        for q in g.query:
            collection_name = q['collection']
            collection_path = os.path.join(collection_root, collection_name)
            all_paras.extend(LambdaRank(collection_path).gen_lambdarank_paras( methods ) )

    #print all_paras
    gen_batch_framework('run_lambdarank', 'l2', all_paras)


def run_lambdarank(para_file):
    with open(para_file) as f:
        reader = csv.reader(f)
        for row in reader:
            collection_path = row[0]
            qid = row[1]
            method_name = row[2]
            method_paras = row[3]
            output_fn = row[4]
            LambdaRank(collection_path).process(qid, method_name, method_paras, output_fn)

def print_lambdarank(print_details=False):
    for c in g.query:
        r = LambdaRank(os.path.join(collection_root, c['collection']))
        print '-'*40
        print c['collection']
        print '-'*40
        r.print_results(print_details)

def print_para_lambdarank(method):
    for c in g.query:
        r = LambdaRank(os.path.join(collection_root, c['collection']))
        print '-'*40
        print c['collection']
        print '-'*40
        r.print_results_para(method)


def gen_ranknet_batch():
    all_paras = []
    with open('lambdarank.json') as f:
        methods = json.load(f)['methods']
        for q in g.query:
            collection_name = q['collection']
            collection_path = os.path.join(collection_root, collection_name)
            all_paras.extend(RankNet(collection_path).gen_lambdarank_paras( methods ) )

    #print all_paras
    gen_batch_framework('run_ranknet', 'r2', all_paras)


def run_ranknet(para_file):
    with open(para_file) as f:
        reader = csv.reader(f)
        for row in reader:
            collection_path = row[0]
            qid = row[1]
            method_name = row[2]
            method_paras = row[3]
            output_fn = row[4]
            RankNet(collection_path).process(qid, method_name, method_paras, output_fn)

def print_ranknet(print_details=False):
    for c in g.query:
        r = RankNet(os.path.join(collection_root, c['collection']))
        print '-'*40
        print c['collection']
        print '-'*40
        r.print_results(print_details)

def print_para_ranknet(method):
    for c in g.query:
        r = RankNet(os.path.join(collection_root, c['collection']))
        print '-'*40
        print c['collection']
        print '-'*40
        r.print_results_para(method)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-1', '--plot_tfc_constraints', nargs='+',
                       help='plot the relevant document distribution. \
                       Take TF as an example and we start from the simplest case when \
                       |Q|=1.  We could leverage an existing collection \
                       and estimate P( c(t,D)=x | D is a relevant document), \
                       where x = 0,1,2,...maxTF(t). ')
    parser.add_argument('-121', '--gen_plot_tf_rel_batch', nargs='+',
                       help='plot P( D is a relevant document | f(tf,doclen)=x ), \
                       args: [query_len(0 for all queries)] [method_name(method_with_para)] \
                       [method_para_str(string, this is for finding the evaluation results)] \
                       [plot_ratio(boolean)] [avg_or_total(boolean, only if the plot_ratio is false)] \
                       [rel_or_all(boolean, only if the plot_ratio is false)] \
                       [performance_as_legend(boolean)] \
                       [drawline(boolean)] [plotbins(boolean)] [numbins(int)] \
                       [xlimit(float)] [ylimit(float)] [zoom_x(int)] [compact_x(boolean)] \
                       [curve_fitting(boolean)] \
                       [draw_individual(boolean, draw the plot for each query)] \
                       [draw_all(boolean, draw the plot for the collection)] \
                       [output_format(eps|png)]')
    parser.add_argument('-122', '--plot_tf_rel_atom', nargs=1, help='')
    parser.add_argument('-120', '--plot_tf_rel_all', nargs='+',
                       help='plot for all collections (a combined plot)')

    parser.add_argument('-syc1', '--plot_synthetic', nargs='+',
                       help='plot P( D is a relevant document | c(t,D)=x ), \
                       where x = 0,1,2,...maxTF(t) for the synthetic data set \
                       args: [maxTF(int)] [scale_factor(int)] [output_format(eps|png)]')
    parser.add_argument('-syc2', '--plot_synthetic_impact', nargs='+',
                       help='plot the impact of changing the number of relevant \
                       documents for each TF data point \
                       args: [maxTF(int)] [rel_docs_change(int)]')
    parser.add_argument('-syc3', '--cal_interpolation_map', nargs='+',
                       help='calculate the MAP based on interpolation \
                       args: [maxTF(int)] [interpolation_type(int)] [subtype(int)] [output_format(eps|png)] [other_options]')

    parser.add_argument('-2', '--gen_ranking_list', nargs='+',
                       help='input: \
                       [-name] each method should have a name. the name will be added a prefix "hypothesis_stq_" \
                       [-callback_code] each method should have a callback function. we use integer to refer to different callback functions. \
                              the callback functions are defined in hypothesis.py \
                       [-type] each method should have a type. the type decides the implementation. \
                              e.g. we may have multiple implementations of TF functions. \
                       [-other_paras] we can add arbitry number of parameters after the name and type')


    parser.add_argument('-21', '--gen_perfect_ranking_list', action='store_true',
                       help='')

    parser.add_argument('-2p', '--print_eval', nargs='+',
                       help='print the evaluation results for methods list. INPUT: [methods list]')

    parser.add_argument('-pp1', '--print_best', nargs='+',
                       help='print the optimal performances and the paras for the methods. INPUT: [methods list]')
    parser.add_argument('-pp2', '--print_statistics', nargs='+',
                       help='print the statistics of collection')
    parser.add_argument('-pp3', '--print_map_with_cut_maxTF', nargs=1,
                       help='print MAP for cut maxTF')

    parser.add_argument('-g1', '--gen_doc_details_batch', action='store_true',
                       help='Generate the document details for single term queries')
    parser.add_argument('-g2', '--gen_doc_details_atom', nargs=1,
                       help='Generate the document details for single term queries')
    

    parser.add_argument('-l1', '--lambdarank_batch', action='store_true',
                       help='LambdaRank related. This is to get the optimal parameters for classic models')
    parser.add_argument('-l2', '--lambdarank_atom', nargs=1,
                       help='LambdaRank related. This is to get the optimal parameters for classic models')
    parser.add_argument('-lp', '--lambdarank_print', nargs='?',
                       help='Print the optimal performances of lambdarank')
    parser.add_argument('-lpp', '--lambdarank_print_para', nargs=1,
                       help='Print the optimal performances of lambdarank')

    parser.add_argument('-r1', '--ranknet_batch', action='store_true',
                       help='Ranknet related. This is to get the optimal parameters for classic models')
    parser.add_argument('-r2', '--ranknet_atom', nargs=1,
                       help='Ranknet related. This is to get the optimal parameters for classic models')
    parser.add_argument('-rp', '--ranknet_print', nargs='?',
                       help='Print the optimal performances of ranknet')
    parser.add_argument('-rpp', '--ranknet_print_para', nargs=1,
                       help='Print the optimal performances of ranknet')

    parser.add_argument('-s1', '--svmmap_data', action='store_true',
                       help='Output the data files for SVMMAP')


    args = parser.parse_args()
    collection_root = '../../../reproduce/collections/'

    if args.plot_tfc_constraints:
        PlotRelTF().plot_tfc_constraints(args.plot_tfc_constraints)

    if args.gen_plot_tf_rel_batch:
        gen_plot_tf_rel_batch(args.gen_plot_tf_rel_batch)
    if args.plot_tf_rel_atom:
        plot_tf_rel_atom(args.plot_tf_rel_atom[0])
    if args.plot_tf_rel_all:
        plot_tf_rel_all(args.plot_tf_rel_all)

    if args.plot_synthetic:
        PlotSyntheticMAP().plot(
            int(args.plot_synthetic[0]), 
            int(args.plot_synthetic[1]), 
            args.plot_synthetic[2])
    if args.plot_synthetic_impact:
        PlotSyntheticMAP().plot_num_rel_docs_impact(
            int(args.plot_synthetic_impact[0]), 
            int(args.plot_synthetic_impact[1]),
            args.plot_synthetic_impact[2])

    if args.cal_interpolation_map:
        PlotSyntheticMAP().cal_map_with_interpolation(
            int(args.cal_interpolation_map[0]), 
            int(args.cal_interpolation_map[1]),
            int(args.cal_interpolation_map[2]),
            args.cal_interpolation_map[3],
            args.cal_interpolation_map[4:])

    if args.gen_ranking_list:
        method_name = args.gen_ranking_list[0]
        callback_code = int(args.gen_ranking_list[1])
        method_type = args.gen_ranking_list[2]
        paras = args.gen_ranking_list[3:] if len(args.gen_ranking_list)>3 else []
        paras.insert(0, method_type)
        for c in g.query:
            h = Hypothesis(os.path.join(collection_root, c['collection']))
            if callback_code == 2:
                _callback = h.hypothesis_tf_ln_function
            h.gen_ranking_list(
              'hypothesis_stq_'+method_name+'_'+method_type, 
              _callback, 
              paras)

    if args.gen_perfect_ranking_list:
        for c in g.query:
            print c['collection']
            h = Hypothesis(os.path.join(collection_root, c['collection']))
            h.gen_perfect_ranking_list()

    if args.print_eval:
        methods = args.print_eval
        for c in g.query:
            h = Hypothesis(os.path.join(collection_root, c['collection']))
            print '-'*40
            print c['collection']
            print '-'*40
            h.print_eval(methods)

    if args.print_best:
        methods = args.print_best
        for c in g.query:
            p = Prints(os.path.join(collection_root, c['collection']))
            print '-'*40
            print c['collection']
            print '-'*40
            p.print_best_performances(methods)

    if args.print_statistics:
        methods = args.print_statistics
        for c in g.query:
            p = Prints(os.path.join(collection_root, c['collection']))
            print '-'*40
            print c['collection']
            print '-'*40
            p.print_statistics(methods)

    if args.print_map_with_cut_maxTF:
        for c in g.query:
            p = Prints(os.path.join(collection_root, c['collection']))
            print '-'*40
            print c['collection']
            print '-'*40
            p.print_map_with_cut_maxTF(int(args.print_map_with_cut_maxTF[0]))

    if args.gen_doc_details_batch:
        gen_doc_details_batch()
    if args.gen_doc_details_atom:
        gen_doc_details(args.gen_doc_details_atom[0])


    if args.gen_ranking_list:
        method_name = args.gen_ranking_list[0]
        callback_code = int(args.gen_ranking_list[1])
        method_type = args.gen_ranking_list[2]
        paras = args.gen_ranking_list[3:] if len(args.gen_ranking_list)>3 else []
        paras.insert(0, method_type)
        for c in g.query:
            h = Hypothesis(os.path.join(collection_root, c['collection']))
            if callback_code == 2:
                _callback = h.hypothesis_tf_ln_function
            h.gen_ranking_list(
              'hypothesis_stq_'+method_name+'_'+method_type, 
              _callback, 
              paras)

    if args.gen_perfect_ranking_list:
        for c in g.query:
            print c['collection']
            h = Hypothesis(os.path.join(collection_root, c['collection']))
            h.gen_perfect_ranking_list()

    if args.print_eval:
        methods = args.print_eval
        for c in g.query:
            h = Hypothesis(os.path.join(collection_root, c['collection']))
            print '-'*40
            print c['collection']
            print '-'*40
            h.print_eval(methods)

    if args.print_best:
        methods = args.print_best
        for c in g.query:
            p = Prints(os.path.join(collection_root, c['collection']))
            print '-'*40
            print c['collection']
            print '-'*40
            p.print_best_performances(methods)

    if args.print_statistics:
        methods = args.print_statistics
        for c in g.query:
            p = Prints(os.path.join(collection_root, c['collection']))
            print '-'*40
            print c['collection']
            print '-'*40
            p.print_statistics(methods)

    if args.print_map_with_cut_maxTF:
        for c in g.query:
            p = Prints(os.path.join(collection_root, c['collection']))
            print '-'*40
            print c['collection']
            print '-'*40
            p.print_map_with_cut_maxTF(int(args.print_map_with_cut_maxTF[0]))

    if args.lambdarank_batch:
        gen_lambdarank_batch()
    if args.lambdarank_atom:
        run_lambdarank(args.lambdarank_atom[0])
    if args.lambdarank_print:
        print_lambdarank(args.lambdarank_print[0] != '0')
    if args.lambdarank_print_para:
        print_para_lambdarank(args.lambdarank_print_para[0])

    if args.ranknet_batch:
        gen_ranknet_batch()
    if args.ranknet_atom:
        run_ranknet(args.ranknet_atom[0])
    if args.ranknet_print:
        print_ranknet(args.ranknet_print[0] != '0')
    if args.ranknet_print_para:
        print_para_ranknet(args.ranknet_print_para[0])

    if args.svmmap_data:
        for c in g.query:
            s = SVMMAP(os.path.join(collection_root, c['collection']))
            print '-'*40
            print c['collection']
            print '-'*40
            s.output_data_file()
