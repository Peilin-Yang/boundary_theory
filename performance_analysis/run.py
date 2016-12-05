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

import inspect
from inspect import currentframe, getframeinfo

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils/'))

import g
import ArrayJob
from analysis import PerformaceAnalysis
from prints import Prints

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


def gen_plot_para_trending_batch_paras(outfmt):
    all_paras = []
    for q in g.query:
        collection_name = q['collection']
        collection_path = os.path.join(g.collection_root, collection_name)
        for query_part in q['qf_parts']:
            for metric in q['metrics']:
                all_paras.append((collection_path, query_part, metric, outfmt))

    #print all_paras
    gen_batch_framework('plot_para_trending', '12', all_paras)


def plot_para_trending(para_file):
    with open(para_file) as f:
        reader = csv.reader(f)
        for row in reader:
            collection_path = row[0]
            query_part = row[1]
            metric = row[2]
            outfmt = row[3]
            PerformaceAnalysis().plot_para_trending(collection_path, query_part, metric, outfmt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-11', '--gen_plot_para_trending_batch_paras', 
        nargs=1,
        help='put the output format as the parameter, e.g. eps or png')
    parser.add_argument('-12', '--plot_para_trending', 
        nargs='+',
        help='')

    args = parser.parse_args()

    if args.gen_plot_para_trending_batch_paras:
        gen_plot_para_trending_batch_paras(args.gen_plot_para_trending_batch_paras[0])
    if args.plot_para_trending:
        plot_para_trending(args.plot_para_trending[0])


