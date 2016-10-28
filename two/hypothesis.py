# -*- coding: utf-8 -*-
import sys,os
import math
import re
import argparse
import json
import ast
import copy
from subprocess import Popen, PIPE
from operator import itemgetter

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils/'))
from base import SingleQueryAnalysis
from collection_stats import CollectionStats
from query import Query
from judgment import Judgment
from evaluation import Evaluation
from gen_doc_details import GenDocDetails

import numpy as np
import scipy.stats


class Hypothesis(object):
    """
    Generate some results for our Hypothesis.
    """

    def __init__(self, path):
        super(Hypothesis, self).__init__()
        self.collection_path = os.path.abspath(path)
        if not os.path.exists(self.collection_path):
            frameinfo = getframeinfo(currentframe())
            print frameinfo.filename, frameinfo.lineno
            print '[TieBreaker Constructor]:Please provide a valid collection path'
            exit(1)

        self.detailed_doc_stats_folder = os.path.join(self.collection_path, 'detailed_doc_stats')
        self.results_folder = os.path.join(self.collection_path, 'merged_results')
        self.eval_folder = os.path.join(self.collection_path, 'evals')

    def cal_map(self, ranking_list, has_total_rel=False, total_rel=-1):
        if not has_total_rel:
            total_rel = 0
        cur_rel = 0
        s = 0.0
        for i, ele in enumerate(ranking_list):
            docid = ele[0]
            rel = int(ele[1])>=1
            if rel:
                cur_rel += 1
                s += cur_rel*1.0/(i+1)
                if not has_total_rel:
                    total_rel += 1
        #print s/total_rel
        return s/total_rel


    def hypothesis_tf_function(self, tf, _type=2, scale=1.0, mu=0.0, sigma=1.0):
        """
        The Hypothesis for the TF function of disk45 collection.
        The function is :
        S = N (μ, σ )(μ = 5, σ = 2), if tf ≤ 10
        S = 1, if 10 < tf ≤ 20
        S = tf, if tf > 20
        """
        if _type == 1:
            if tf <= 10:
                return scipy.stats.norm(9, 2).pdf(tf)
            elif tf > 10 and tf <= 20:
                return 0.1+scipy.stats.norm(18, 2).pdf(tf)
            elif tf > 20:
                return tf
        elif _type == 2:
            if tf <= 20:
                return scale+scipy.stats.norm(mu, sigma).pdf(tf)
            elif tf > 20:
                return tf

    def hypothesis_tf_ln_function(self, paras=[]):
        """
        The Hypothesis for the TF-LN function.
        """
        tf, ln, avdl, ctf, total_terms, idf = tuple(paras[1:])
        _type = int(paras[0])
        if _type == 1:
            return tf*1.0/ln
        elif _type == 2:
            return tf*1.0/math.log(ln)
        elif _type == 3:
            return math.log(tf*1.0)/(tf+math.log(ln))
        elif _type == 4:
            return math.log(tf+1.0)/(ln+1000)
        elif _type == 5:
            if 'gov2' in self.collection_path:
                delta = 3
            elif 'disk12' in self.collection_path:
                delta = 0.05
            elif 'disk45' in self.collection_path:
                delta = 0.0
            elif 'wt2g' in self.collection_path:
                delta = 0.0
            return (math.log(tf*1.0)+delta)/math.log(ln)
        elif _type == 6:
            return tf*1.0/(tf+math.log(ln))
        elif _type == 7:
            return math.log(tf*1.0)/(math.log(tf*1.0)+math.log(ln))
        elif _type == 8:
            return math.log(tf*1.0)/math.log(ln)
        elif _type == 9:
            return math.log(tf*1.0)/(math.log(tf*1.0)+math.log(ln*1.0/avdl))
        elif _type == 10:
            return math.log(tf*1.0)/(tf+math.log(ln*1.0/avdl))
        elif _type == 11:
            return math.log(tf*1.0)/(tf+ln*1.0/avdl)

    def output_results(self, res, method):
        with open(os.path.join(self.results_folder, 'title-method:%s'%method), 'wb') as output:
            for qid in res:
                res[qid].sort(key=itemgetter(1), reverse=True)
                for ele in res[qid]:
                    output.write('%s Q0 %s 1 %s 0\n' % (qid, ele[0], ele[1]))        

    def eval(self, method):
        qrel_program='trec_eval -m all_trec -q'.split()
        result_file_path=os.path.join(self.results_folder, 'title-method:%s'%method)
        eval_file_path=os.path.join(self.eval_folder, 'title-method:%s'%method)
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

    def print_map(self, fn):
        with open(fn) as f:
            j = json.load(f)
            print j['all']['map']

    def print_eval(self, methods=[]):
        for fn in os.listdir(self.eval_folder):
            for method in methods:
                if 'title-method:hypothesis_stq_%s'%method in fn:
                    print fn,
                    self.print_map(os.path.join(self.eval_folder, fn))
                    break


    def gen_ranking_list(self, method, _callback, paras):
        """
        We get the statistics from /collection_path/detailed_doc_stats/ 
        so that we can get everything for the top 10,000 documents for 
        each query generated by Dirichlet language model method.
        """
        single_queries = Query(self.collection_path).get_queries_of_length(1)
        queries = {ele['num']:ele['title'] for ele in single_queries}
        doc_details = GenDocDetails(self.collection_path)
        cs = CollectionStats(self.collection_path)
        avdl = cs.get_avdl()
        total_terms = cs.get_total_terms()
        res = {}
        for qid in queries:
            print queries[qid]
            res[qid] = []
            idx = 0
            ctf = cs.get_term_collection_occur(queries[qid])
            idf = cs.get_term_logidf1(queries[qid])
            #for row in cs.get_qid_details(qid):
            for row in doc_details.get_qid_details(qid):
                docid = row['docid']
                total_tf = float(row['total_tf'])
                doc_len = float(row['doc_len'])
                localpara = copy.deepcopy(paras)
                localpara.extend([total_tf, doc_len, avdl, ctf, total_terms, idf])
                score = _callback(localpara)
                res[qid].append((docid, score))
                idx += 1
                if idx >= 1000:
                    break
        self.output_results(res, method)
        self.eval(method)


    def _sort_by_map(self, ele):
        if not ele[1]:
            return 0
        ele[1].sort(key=itemgetter(1, 0), reverse=True)
        ranking_list = [(e[0], e[2]) for e in ele[1]]
        s = self.cal_map(ranking_list, True, ele[1][0][3])
        s += ele[0]
        return s

    def gen_perfect_ranking_list(self, plotbins=True, numbins=60):
        cs = CollectionStats(self.collection_path)
        single_queries = Query(self.collection_path).get_queries_of_length(1)
        queries = {ele['num']:ele['title'] for ele in single_queries}
        #print qids
        rel_docs = Judgment(self.collection_path).get_relevant_docs_of_some_queries(queries.keys(), 1, 'dict')
        idfs = [(qid, math.log(cs.get_term_IDF1(queries[qid]))) for qid in rel_docs]
        idfs.sort(key=itemgetter(1))
        res = {}
        for qid,idf in idfs:
            x_dict = {}
            res[qid] = []
            score_mapping = {}
            maxScore = -99999999
            for row in cs.get_qid_details(qid):
                docid = row['docid']
                total_tf = float(row['total_tf'])
                doc_len = float(row['doc_len'])
                rel_score = int(row['rel_score'])
                score = math.log(total_tf+1.0)/(math.log(total_tf+1.0)+math.log(doc_len))
                #score = total_tf/(total_tf + doc_len)
                score_mapping[docid] = score
                if score > maxScore:
                    maxScore = score
                rel = (rel_score>=1)
                if score not in x_dict:
                    x_dict[score] = [0, 0, [docid, score, rel, len(rel_docs[qid])]] # [rel_docs, total_docs]
                if rel:
                    x_dict[score][0] += 1
                x_dict[score][1] += 1

            # xaxis = x_dict.keys()
            # xaxis.sort()
            # yaxis = [(x_dict[x][0]*1./x_dict[x][1], x_dict[x][2]) for x in xaxis]
            # if plotbins:
            interval = maxScore*1.0/numbins
            newxaxis = [i for i in np.arange(0, maxScore+1e-10, interval)]
            newyaxis = [[0.0, 0.0, []] for x in newxaxis]
            for x in x_dict:
                newx = int(x / interval)
                # print x_dict[x]
                newyaxis[newx][0] += x_dict[x][0]
                newyaxis[newx][1] += x_dict[x][1]
                newyaxis[newx][2].append( x_dict[x][2] )
                # print x, newx
                # print newxaxis
                # print newyaxis
                # raw_input()
            xaxis = newxaxis
            yaxis = [(ele[0]*1.0/ele[1], ele[2]) if ele[1] != 0 else (0, []) for ele in newyaxis]
            yaxis.sort(key=itemgetter(0), reverse=True)
            #yaxis.sort(key=self._sort_by_map, reverse=True)
            sbase = 1e9
            for ele in yaxis:
                for doc in ele[1]:
                    docid = doc[0]  
                    if len(res[qid]) < 1000:     
                        res[qid].append((docid, sbase+score_mapping[docid]))
                    sbase -= 100

            #print len(res[qid])

        method = 'hypothesis_stq_tf_ln_upperbound'
        self.output_results(res, method)
        self.eval(method)

