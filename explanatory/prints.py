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
from collection_stats import CollectionStats
from query import Query
from judgment import Judgment
from evaluation import Evaluation
from performance import Performances
from gen_doc_details import GenDocDetails
from rel_tf_stats import RelTFStats

import numpy as np
import scipy.stats


class Prints(object):
    """
    Prints all kinds of information
    """

    def __init__(self, corpus_path, corpus_name):
        self.collection_path = os.path.abspath(corpus_path)
        if not os.path.exists(self.collection_path):
            print '[Evaluation Constructor]:Please provide valid corpus path'
            exit(1)

        self.collection_name = corpus_name
        self.all_results_root = '../../all_results'
        if not os.path.exists(self.all_results_root):
            os.path.makedirs(self.all_results_root)

        self.detailed_doc_stats_folder = os.path.join(self.collection_path, 'detailed_doc_stats')
        self.results_folder = os.path.join(self.collection_path, 'merged_results')
        self.eval_folder = os.path.join(self.collection_path, 'evals')

    def print_best_performances(self, methods=[]):
        single_queries = Query(self.collection_path).get_queries_of_length(1)
        queries = {ele['num']:ele['title'] for ele in single_queries}
        cs = CollectionStats(self.collection_path)
        performance = Performances(self.collection_path)
        res = performance.gen_optimal_performances_queries(methods, queries.keys())
        print res


    def print_statistics(self, methods):
        single_queries = Query(self.collection_path).get_queries_of_length(1)
        queries = {ele['num']:ele['title'] for ele in single_queries}
        cs = CollectionStats(self.collection_path)
        performance = Performances(self.collection_path)
        res = performance.gen_optimal_performances_queries(methods, queries.keys())

        avdl = cs.get_avdl()
        total_terms = cs.get_total_terms()
        collection_freq = []
        for qid in queries:
            idx = 0
            ctf = cs.get_term_collection_occur(queries[qid])
            idf = cs.get_term_logidf1(queries[qid])
            collection_freq.append( ctf*1.0/total_terms )
        print avdl
        print np.mean(collection_freq)

        for ele in res:
            label = ele[0]
            p = ele[1]
            para = float(ele[2].split(':')[1])
            print label
            if 'okapi' in label:
                print 'b:', para, 'beta:', 1.2*para/avdl, 'c2:', 1.2*(1-para)
            if 'pivoted' in label:
                print 's:', para, 'beta:', para/avdl, 'c2:', 1-para

            

    def get_docs_tf(self):
        """
        We get the statistics from /collection_path/detailed_doc_stats/ 
        so that we can get everything for the top 10,000 documents for 
        each query generated by Dirichlet language model method.
        """
        all_queries = Query(self.collection_path).get_queries()
        queries = {ele['num']:ele['title'] for ele in all_queries}
        doc_details = GenDocDetails(self.collection_path)
        res = {}
        for qid in queries:
            res[qid] = []
            idx = 0
            try:
                for row in doc_details.get_qid_details(qid):
                    docid = row['docid']
                    tf = float(row['total_tf'])
                    #score = self.hypothesis_tf_function(tf, _type, scale, mu, sigma)
                    res[qid].append([docid, tf])
                    idx += 1
                    if idx >= 1000:
                        pass
                res[qid].sort(key=itemgetter(1,0), reverse=True)
            except IOError:
                pass
        return res

    def cut_docs_tf_with_maxTF(self, maxTF=20):
        docs_tf = self.get_docs_tf()
        dropped_tf = {}
        dropped_list = []
        for qid in docs_tf:
            tf = [ele for ele in docs_tf[qid] if ele[1] <= maxTF]
            tf.sort(key=itemgetter(1,0), reverse=True)
            dropped = [ele for ele in docs_tf[qid] if ele[1] > maxTF]
            dropped_list.append(len(dropped))
            dropped_tf[qid] = dropped
            docs_tf[qid] = tf
        print 'avg dropped:', np.mean(np.asarray(dropped_list))
        return docs_tf, dropped_tf

    def cal_map(self, ranking_list_with_judgement):
        cur_rel = 0
        s = 0.0
        total = 0
        for i, ele in enumerate(ranking_list_with_judgement):
            docid = ele[0]
            is_rel = ele[1]
            if is_rel:
                cur_rel += 1
                total += 1
                s += cur_rel*1.0/(i+1)
        if total == 0:
            return 0
        return s/total

    def print_map_with_cut_maxTF(self, maxTF=20, type=1):
        single_queries = Query(self.collection_path).get_queries_of_length(1)
        rel_docs = Judgment(self.collection_path).get_relevant_docs_of_some_queries([ele['num'] for ele in single_queries])
        cutted_docs, dropped_tf = self.cut_docs_tf_with_maxTF(maxTF)
        maps = []
        rel_smaller_than_maxTF = []
        rel_larger_than_maxTF = []
        for qid in cutted_docs:
            if qid not in rel_docs or not rel_docs[qid]:
                continue
            ranking_with_judge = [(doc[0], doc[0] in [ele[0] for ele in rel_docs[qid]]) for doc in cutted_docs[qid]]
            rel_smaller_cnt = len([ele for ele in ranking_with_judge if ele[1]])
            rel_smaller_than_maxTF.append(rel_smaller_cnt)
            larger_ones = [(doc[0], doc[0] in [ele[0] for ele in rel_docs[qid]]) for doc in dropped_tf[qid]]
            rel_larger_cnt = len([ele for ele in larger_ones if ele[1]])
            rel_larger_than_maxTF.append(rel_larger_cnt)
            maps.append(self.cal_map(ranking_with_judge))
        print np.mean(np.asarray(maps))
        print 'rel_smaller:', np.mean(np.asarray(rel_smaller_than_maxTF)),
        print 'rel_larger:', np.mean(np.asarray(rel_larger_than_maxTF)),
        print 'ratio:', np.mean(np.asarray(rel_larger_than_maxTF)) / np.mean(np.asarray(rel_smaller_than_maxTF))

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

    def okapi_apply(self, tf, idf, doclen, avdl, b):
        k1 = 1.2
        return (k1+1.0)*tf/(tf+k1*(1-b+b*doclen*1.0/avdl))*idf

    def okapi(self, data, b=0.25):
        tfs = data[1]
        dfs = data[2]
        doclens = data[3]
        rels = data[4]
        cs = CollectionStats(self.collection_path)
        idfs = np.reshape(np.repeat(np.log((cs.get_doc_counts() + 1)/(dfs+1e-4)), tfs.shape[1]), tfs.shape)
        avdl = cs.get_avdl()
        print tfs, dfs, doclens
        print tfs.shape
        print b
        #r = np.apply_along_axis(self.okapi_apply, 0, tfs, idfs, doclens, avdl, b)
        k1 = 1.2
        r = (k1+1.0)*tfs/(tfs+k1*(1-b+b*doclens*1.0/avdl))*idfs
        print r
        exit()
        return np.sum(r, axis=0)

    def print_ranking_using_doc_details_file(self, query_length=2, model='okapi'):
        model_mapping = {
            'okapi': self.okapi
        }
        query_length = int(query_length)
        doc_details = self.read_docdetails_data(query_length)
        rel_data = self.read_rel_data(query_length)
        ranking_lists = {}
        for qid in doc_details:
            print qid
            ranking_lists[qid] = model_mapping[model](doc_details[qid], 
                float(rel_data[qid]['AP'][model][2].split(':')[1]))
            print ranking_lists[qid]


        
