# -*- coding: utf-8 -*-
import sys,os
import csv
import json
import re
import string
import ast
import uuid
import itertools
import codecs
from operator import itemgetter
from subprocess import Popen, PIPE
from inspect import currentframe, getframeinfo
import argparse

import numpy as np

from performance import Performances
from collection_stats import CollectionStats

class RunProximitySubqueries(RunSubqueries):
    """
    run the proximity sub-queries.
    """
    def __init__(self, path, corpus_name):
        super(RunProximitySubqueries, self).__init__(path, corpus_name)
        self.subqueries_runfiles_root = os.path.join(self.output_root, 'proximity_runfiles')
        if not os.path.exists(self.subqueries_runfiles_root):
            os.makedirs(self.subqueries_runfiles_root)
        self.subqueries_performance_root = os.path.join(self.output_root, 'proximity_performances')
        if not os.path.exists(self.subqueries_performance_root):
            os.makedirs(self.subqueries_performance_root)
        self.collected_results_root = os.path.join(self.output_root, 'proximity_collected_results')
        if not os.path.exists(self.collected_results_root):
            os.makedirs(self.collected_results_root)

        self.final_results_root = '../all_results'
        self.final_output_root = os.path.join(self.final_results_root, 'subqueries')
        if not os.path.exists(self.final_output_root):
            os.makedirs(self.final_output_root)

        type_mapping = {
            1: 'uw',
            2: 'od',
            3: 'uw+od'
        }

    def batch_run_subqueries_paras(self, _type=1, query_length=0):
        all_paras = []
        if query_length == 0: #all queries
            queries = self.get_queries()
        else:
            queries = self.get_queries_of_length(query_length)
        queries = {ele['num']:ele['title'] for ele in queries}
        for qid, query in queries.items():
            all_subqueries = self.get_subqueries(query)
            if not os.path.exists(os.path.join(self.subqueries_mapping_root, qid)):
                with open(os.path.join(self.subqueries_mapping_root, qid), 'wb') as f:
                    json.dump(all_subqueries, f, indent=2)
            for subquery_id, subquery_str in all_subqueries.items():
                subquery_len = len(subquery_str.split())
                if _type == 1:
                    subquery_str = '#uw%d(' % (4*subquery_len)+subquery_str+')'
                elif _type == 2:
                    subquery_str = '#od%d(' % (4*subquery_len)+subquery_str+')'
                elif _type == 3:
                    subquery_str = '#combine('
                        + '#uw%d(' % (4*subquery_len)+subquery_str+')' 
                        + '#od%d(' % (4*subquery_len)+subquery_str+')' 
                        +')'
                type_str = type_mapping[_type]
                run_file_root = os.path.join(self.subqueries_runfiles_root, type_str)
                if not os.path.exists(run_file_root):
                    os.makedirs(run_file_root)
                performance_root = os.path.join(self.subqueries_performance_root, type_str)
                if not os.path.exists(performance_root):
                    os.makedirs(performance_root)
                runfile_fn = os.path.join(run_file_root, qid+'_'+subquery_id)
                performance_fn = os.path.join(performance_root, qid+'_'+subquery_id)
                if not os.path.exists(performance_fn):
                    all_paras.append((self.corpus_path, self.collection_name, qid, subquery_str, subquery_id, type_str, runfile_fn, performance_fn))
        return all_paras

    def run_subqueries(self, qid, subquery_id, query, type_str, runfile_ofn, eval_ofn):
        self.run_indri_runquery(query, runfile_ofn, qid, type_str)
        self.eval(runfile_ofn, eval_ofn)

    def batch_collect_results_paras(self):
        queries = self.get_queries()
        return [(self.corpus_path, self.collection_name, q['num']) for q in queries]

    def collection_all_results(self, req_qid):
        qid_results = []
        for fn in os.listdir(self.subqueries_performance_root):
            fn_split = fn.split('_')
            qid = fn_split[0]
            if qid != req_qid:
                continue
            subquery_id = fn_split[1]
            model_para = fn_split[2]
            with open(os.path.join(self.subqueries_mapping_root, qid)) as f:
                subquery_mapping = json.load(f)
            try:
                with open(os.path.join(self.subqueries_performance_root, fn)) as f:
                    first_line = f.readline()
                    ap = first_line.split()[-1]
            except:
                continue
            #print fn, subquery_mapping
            qid_results.append( (subquery_id, subquery_mapping[subquery_id], model_para, ap) )

        qid_results.sort(key=self.sort_subquery_id)
        with open(os.path.join(self.collected_results_root, req_qid), 'wb') as f:
            wr = csv.writer(f)
            wr.writerows(qid_results)


    def sort_subquery_key(self, k):
        subquery_id_split = k.split('_')[0].split('-')
        return int(subquery_id_split[0])+float(subquery_id_split[1])/10.0

    def output_optimal_dist(self):
        optimals = {}
        queries = self.get_queries()
        queries = {ele['num']:ele['title'] for ele in queries}
        model_paras = set()
        for qid in os.listdir(self.collected_results_root):
            with open(os.path.join(self.collected_results_root, qid)) as f:
                csvr = csv.reader(f)
                for row in csvr:
                    subquery_id = row[0]
                    subquery = row[1]
                    model_para = row[2]
                    model_paras.add(model_para)
                    ap = float(row[3])
                    if model_para not in optimals:
                        optimals[model_para] = {}
                    subquery_len = int(subquery_id.split('-')[0])
                    if qid not in optimals[model_para]:
                        optimals[model_para][qid] = []
                    optimals[model_para][qid].append((subquery_len, ap))
        for model_para in optimals:
            for qid in optimals[model_para]:
                optimals[model_para][qid].sort(key=itemgetter(1), reverse=True)

        res = {}
        for qid, query in queries.items():
            query_len = len(query.split())
            for model_para in model_paras:
                if qid not in optimals[model_para]:
                    continue
                if model_para not in res:
                    res[model_para] = {}
                if query_len not in res[model_para]:
                    res[model_para][query_len] = {}
                optimal_subquery_len = optimals[model_para][qid][0][0]
                if optimal_subquery_len not in res[model_para][query_len]:
                    res[model_para][query_len][optimal_subquery_len] = 0
                res[model_para][query_len][optimal_subquery_len] += 1

        with open(os.path.join(self.final_output_root, self.collection_name+'-optimal_subquery_dist.md'), 'wb') as f:
            f.write('## %s\n' % (self.collection_name))
            for model_para in res:
                f.write('\n### %s\n' % (model_para))
                f.write('| | | | | |\n')
                f.write('|--------|--------|--------|--------|--------|\n')
                for query_len in res[model_para]:
                    f.write('| %d |' % (query_len))
                    for optimal_subquery_len in res[model_para][query_len]:
                        f.write(' %d:%d |' % (optimal_subquery_len, res[model_para][query_len][optimal_subquery_len]))
                    f.write('\n')


    def output_results(self, query_length=0):
        if query_length == 0: #all queries
            queries = self.get_queries()
        else:
            queries = self.get_queries_of_length(query_length)
        queries = {int(ele['num']):ele['title'] for ele in queries}
        methods = ['okapi', 'dir']
        optimal_model_performances = Performances(self.corpus_path).load_optimal_performance(methods)
        model_paras = []
        for p in optimal_model_performances:
            model_para = 'method:%s,' % p[0] + p[2]
            model_paras.append(model_para)
        subquery_data = {}
        for qid, query in queries.items():
            subquery_data[qid] = {}
            with open(os.path.join(self.collected_results_root, str(qid))) as f:
                csvr = csv.reader(f)
                for row in csvr:
                    subquery_id = row[0]
                    subquery = row[1]
                    model_para = row[2]
                    ap = row[3]
                    _key = subquery_id+'_'+subquery
                    if _key not in subquery_data[qid]:
                        subquery_data[qid][_key] = {}
                    subquery_data[qid][_key][model_para] = ap

        all_data = []
        for qid in sorted(subquery_data):
            subqueries = sorted(subquery_data[qid], key=self.sort_subquery_key)
            all_data.append(subqueries)
            all_data[-1].insert(0, str(qid))
            for model_para in model_paras:
                all_data.append([str(subquery_data[qid][q][model_para]) if model_para in subquery_data[qid][q] else '' for q in subqueries[1:]])
                all_data[-1].insert(0, model_para)

        with open(os.path.join(self.final_output_root, self.collection_name+'-'+str(query_length)+'.csv'), 'wb') as f:
            cw = csv.writer(f)
            cw.writerows(all_data)

        # markdown output
        cs = CollectionStats(self.corpus_path)
        max_row_len = max([len(ele) for ele in all_data])
        with open(os.path.join(self.final_output_root, self.collection_name+'-'+str(query_length)+'.md'), 'wb') as f:
            f.write('%s|\n' % ('| ' * max_row_len))
            f.write('%s|\n' % ('|---' * max_row_len))
            cur_qid = 0
            cur_queries = []
            for i, data in enumerate(all_data):
                if i % 3 != 0: # numerical values (MAP) line
                    if len(data) == 1:
                        continue
                    _max = np.argmax([float(ele) for ele in data[1:]])+1
                    for j in range(len(data)):
                        if j == _max:
                            f.write('| **%s** ' % data[j])
                        else:
                            f.write('| %s ' % data[j])
                    f.write(' |\n')
                    # read and output the details in runfile
                    if _max+1 != len(data):
                        lines_cnt = 10
                        runfile_max = os.path.join(self.subqueries_runfiles_root, '%s_%s_%s' % (cur_qid, cur_queries[_max].split('_')[0], data[0]))
                        with open(runfile_max) as rf:
                            first_few_lines_max = [line.split()[1]+' '+line.split()[-1] for line in rf.readlines()[:lines_cnt]]
                        runfile_allterms = os.path.join(self.subqueries_runfiles_root, '%s_%s_%s' % (cur_qid, cur_queries[-1].split('_')[0], data[0]))
                        with open(runfile_allterms) as rf:
                            first_few_lines_allterms = [line.split()[1]+' '+line.split()[-1] for line in rf.readlines()[:lines_cnt]]
                        # also read term stats
                        all_terms_stats = []
                        for t in cur_queries[-1].split('_')[1].split():
                            all_terms_stats.append([k+':'+str(v) for k,v in cs.get_term_stats(t).items() if k!='raw'])
                            all_terms_stats[-1].insert(0, t)
                        term_line_idx = 0
                        for zipped in zip(first_few_lines_max, first_few_lines_allterms):
                            for ele in zipped:
                                f.write('| %s ' % (ele))
                            for ele in all_terms_stats:
                                f.write('| %s ' % (ele[term_line_idx]))
                            term_line_idx += 1
                            f.write(' |\n')
                else: # qid query line
                    cur_qid = data[0]
                    cur_queries = data
                    f.write('| %s |\n' % (' | '.join([d.split('_')[-1] for d in data])))
