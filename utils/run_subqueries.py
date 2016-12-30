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
from subprocess import Popen, PIPE
from inspect import currentframe, getframeinfo
import argparse

import numpy as np
import markdown

from performance import Performances
from collection_stats import CollectionStats

class RunSubqueries(object):
    """
    run the sub-queries since they potentially outperform the original query.
    """
    def __init__(self, path, corpus_name):
        self.corpus_path = os.path.abspath(path)
        if not os.path.exists(self.corpus_path):
            frameinfo = getframeinfo(currentframe())
            print frameinfo.filename, frameinfo.lineno
            print '[Query Constructor]:path "' + self.corpus_path + '" is not a valid path'
            print '[Query Constructor]:Please provide a valid corpus path'
            exit(1)

        self.collection_name = corpus_name
        self.query_file_path = os.path.join(self.corpus_path, 'raw_topics')
        if not os.path.exists(self.query_file_path):
            frameinfo = getframeinfo(currentframe())
            print frameinfo.filename, frameinfo.lineno
            print """No query file found! 
                query file should be called "raw_topics" under 
                corpus path. You can create a symlink for it"""
            exit(1)

        self.parsed_query_file_path = os.path.join(self.corpus_path, 'parsed_topics.json')
        self.output_root = os.path.join(self.corpus_path, 'subqueries')
        if not os.path.exists(self.output_root):
            os.makedirs(self.output_root)
        self.subqueries_mapping_root = os.path.join(self.output_root, 'mappings')
        if not os.path.exists(self.subqueries_mapping_root):
            os.makedirs(self.subqueries_mapping_root)
        self.subqueries_runfiles_root = os.path.join(self.output_root, 'runfiles')
        if not os.path.exists(self.subqueries_runfiles_root):
            os.makedirs(self.subqueries_runfiles_root)
        self.subqueries_performance_root = os.path.join(self.output_root, 'performances')
        if not os.path.exists(self.subqueries_performance_root):
            os.makedirs(self.subqueries_performance_root)
        self.collected_results_root = os.path.join(self.output_root, 'collected_results')
        if not os.path.exists(self.collected_results_root):
            os.makedirs(self.collected_results_root)

        self.final_results_root = '../all_results'
        self.final_output_root = os.path.join(self.final_results_root, 'subqueries')
        if not os.path.exists(self.final_output_root):
            os.makedirs(self.final_output_root)

    def get_queries(self):
        """
        Get the query of a corpus

        @Return: a list of dict [{'num':'401', 'title':'the query terms',
         'desc':description, 'narr': narrative description}, ...]
        """
        with open(self.parsed_query_file_path) as f:
            return json.load(f)

    def get_queries_dict(self):
        """
        Get the query of a corpus

        @Return: a dict with keys as qids {'401':{'title':'the title', 'desc':'the desc'}, ...}
        """
        all_queries = self.get_queries()
        all_queries_dict = {}
        for ele in all_queries:
            qid = ele['num']
            all_queries_dict[qid] = ele

        return all_queries_dict
        
    def get_queries_lengths(self, part='title'):
        """
        For a set of queries, return the lengths of the queries

        @Return: a list of integers showing the lengths of the queries
        """
        queries = self.get_queries()
        lengths = set([len(q[part].split()) for q in queries])
        lengths = list(lengths)
        lengths.sort()
        return lengths


    def get_queries_of_length(self, length, part='title'):
        """
        Get the queries of a specific length

        @Input:
            length - the specific length. For example, length=1 get all queries
                     with single term

        @Return: a list of dict [{'num':'403', 'title':'osteoporosis',
         'desc':description, 'narr': narrative description}, ...]
        """

        all_queries = self.get_queries()
        filtered_queries = [ele for ele in all_queries if len(ele[part].split()) == length]

        return filtered_queries

    def run_indri_runquery(self, query_str, runfile_ofn, qid='0', rule=''):
        with open(runfile_ofn, 'w') as f:
            command = ['IndriRunQuery_EX -index=%s -trecFormat=True -count=1000 -query.number=%s -query.text="%s" -rule=%s' 
                % (os.path.join(self.corpus_path, 'index'), qid, query_str, rule)]
            p = Popen(command, shell=True, stdout=f, stderr=PIPE)
            returncode = p.wait()
            p.communicate()
            if returncode != 0:
                raise NameError("Run Query Error: %s" % (command) )

    def eval(self, runfile_ofn, eval_ofn):
        judgment_file = os.path.join(self.corpus_path, 'judgement_file')
        with open(eval_ofn, 'w') as f:
            p = Popen(['trec_eval -m map %s %s' % (judgment_file, runfile_ofn)], shell=True, stdout=f, stderr=PIPE)
            returncode = p.wait()
            p.communicate()

    def get_subqueries(self, query_str):
        """
        return {'subquery string': 'subquery id', ...}
        """
        all_subqueries = {}
        terms = query_str.split()
        for i in range(1, len(terms)+1): # including the query itself
            j = 0
            for ele in itertools.combinations(terms, i):
                all_subqueries['%d-%d' % (i, j)] = ' '.join(ele)
                j += 1
        return all_subqueries

    def batch_run_subqueries_paras(self, query_length=0):
        all_paras = []
        methods = ['okapi', 'dir']
        optimal_model_performances = Performances(self.corpus_path).load_optimal_performance(methods)
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
            for subquery_id, subquery_str  in all_subqueries.items():
                for p in optimal_model_performances:
                    indri_model_para = 'method:%s,' % p[0] + p[2]
                    runfile_fn = os.path.join(self.subqueries_runfiles_root, qid+'_'+subquery_id+'_'+indri_model_para)
                    performance_fn = os.path.join(self.subqueries_performance_root, qid+'_'+subquery_id+'_'+indri_model_para)
                    if not os.path.exists(performance_fn):
                        all_paras.append((self.corpus_path, self.collection_name, qid, subquery_str, subquery_id, indri_model_para, runfile_fn, performance_fn))
        return all_paras

    def run_subqueries(self, qid, subquery_id, query, indri_model_para, runfile_ofn, eval_ofn):
        self.run_indri_runquery(query, runfile_ofn, qid, indri_model_para)
        self.eval(runfile_ofn, eval_ofn)

    def sort_subquery_id(self, result):
        subquery_id = result[0]
        return int(subquery_id.split('-')[0])+float(subquery_id.split('-')[1])/10.0

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
            qid_results.append( (subquery_id, subquery_mapping[subquery_id], model_para, ap) )

        qid_results.sort(key=self.sort_subquery_id)
        with open(os.path.join(self.collected_results_root, req_qid), 'wb') as f:
            wr = csv.writer(f)
            wr.writerows(qid_results)


    def sort_subquery_key(self, k):
        subquery_id_split = k.split('_')[0].split('-')
        return int(subquery_id_split[0])+float(subquery_id_split[1])/10.0

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
        
        with codecs.open(os.path.join(self.final_output_root, self.collection_name+'-'+str(query_length)+'.md'), 'r', encoding='utf-8') as f:
            print markdown.markdown(f.read())