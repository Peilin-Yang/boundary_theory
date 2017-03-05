# -*- coding: utf-8 -*-
import sys,os
import csv
import json
import re
import math
import ast
import uuid
import itertools
import codecs
from operator import itemgetter
import subprocess
from subprocess import Popen, PIPE
from inspect import currentframe, getframeinfo
import argparse

import numpy as np
import scipy.stats
from sklearn.datasets import load_svmlight_file

import lightgbm as lgb

from query import Query
from evaluation import Evaluation
from judgment import Judgment
from performance import Performances
from collection_stats import CollectionStats
from subqueries_learning import SubqueriesLearning
from ArrayJob import ArrayJob

class LGBMLearning(SubqueriesLearning):
    """
    learning the subqueries: features generation, learning, etc.
    """
    def __init__(self, path, corpus_name):
        super(LGBMLearning, self).__init__(path, corpus_name)

    def read_data_from_feature_file(self, fn, integer_label=False):
        orig_rows = []
        data = []
        label = []
        tmp_label = []
        query_data = []
        cur_query = -1
        cur_query_line_cnt = -1
        with open(fn) as f:
            for line in f:
                line = line.strip()
                if line:
                    row = line.split()
                    orig_rows.append(row)
                    row_data = [float(ele.split(':')[-1]) for ele in row[2:-2]] # do not include qid
                    data.append(row_data)
                    qid = row[1].split(':')[1]
                    if qid != cur_query:
                        if cur_query_line_cnt != -1:
                            query_data.append(cur_query_line_cnt)
                            if integer_label:
                                tmptmp_label = [round((ele-min(tmp_label))*4/(max(tmp_label) - min(tmp_label)), 0) for ele in tmp_label]
                                max_cnts = []
                                for i, ele in enumerate(tmptmp_label):
                                    if ele == 4.0:
                                        max_cnts.append(i)
                                if len(max_cnts) > 1:
                                    orig_max_idx = 0
                                    for j in range(1, len(tmp_label)):
                                        if tmp_label[j] > tmp_label[orig_max_idx]:
                                            orig_max_idx = j
                                    for max_cnt in max_cnts:
                                        if max_cnt != orig_max_idx:
                                            tmptmp_label[max_cnt] -= 1
                                tmp_label = tmptmp_label
                            label.extend(tmp_label)
                        cur_query = qid
                        cur_query_line_cnt = 0
                        tmp_label = []
                    tmp_label.append(float(row[0]))
                    cur_query_line_cnt += 1
        
        if cur_query_line_cnt > 0:
            query_data.append(cur_query_line_cnt)

        if integer_label:
            tmptmp_label = [round((ele-min(tmp_label))*4/(max(tmp_label) - min(tmp_label)), 0) for ele in tmp_label]
            max_cnts = []
            for i, ele in enumerate(tmptmp_label):
                if ele == 4.0:
                    max_cnts.append(i)
            if len(max_cnts) > 1:
                orig_max_idx = 0
                for j in range(1, len(tmp_label)):
                    if tmp_label[j] > tmp_label[orig_max_idx]:
                        orig_max_idx = j
                for max_cnt in max_cnts:
                    if max_cnt != orig_max_idx:
                        tmptmp_label[max_cnt] -= 1
            tmp_label = tmptmp_label
        label.extend(tmp_label)          
   
        with open(fn+'.new', 'w') as of:
            for i, row in enumerate(orig_rows):
                of.write('%d %s\n' % (label[i], ' '.join(row[1:])))
        
        return [data, label, query_data]


    def test(self):
        # X_train, y_train = load_svmlight_file('lgbm_test_data.txt')
        # print X_train, y_train
        X_train, y_train, q_train = self.read_data_from_feature_file('lgbm_test_data.txt', integer_label=True)
        #X_test, y_test, q_test = self.read_data_from_feature_file('lgbm_test_data.txt')
        # lgb_model  = lgb.LGBMRanker().fit(X_train, y_train,
        #                          group=q_train,
        #                          eval_set=[(X_test, y_test)],
        #                          eval_group=[q_test],
        #                          eval_at=[10],
        #                          verbose=True,
        #                          callbacks=[lgb.reset_parameter(learning_rate=lambda x: 0.95 ** x * 0.1)])


_root = '../../../reproduce/collections/'
output_root = '../../all_results/'


def gen_batch_framework(para_label, batch_pythonscript_para, all_paras, \
        quote_command=False, memory='2G', max_task_per_node=50000, num_task_per_node=50):

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
    arrayjob_script = ArrayJob()
    arrayjob_script.output_batch_qs_file(batch_script_fn, command, quote_command, True, batch_para_fn, len(all_paras), _memory=memory)
    run_batch_gen_query_command = 'qsub %s' % batch_script_fn
    subprocess.call( shlex.split(run_batch_gen_query_command) )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--test", dest='test',
        action='store_true',
        required=False, 
        help="generate the batch run parameters for running the baseline method")

    args = parser.parse_args()

    if args.test:
        LGBMLearning('.', 'test').test()