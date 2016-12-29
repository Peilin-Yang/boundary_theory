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
import scipy.stats
import markdown

from performance import Performances
from collection_stats import CollectionStats
from run_subqueries import RunSubqueries

class SubqueriesLearning(RunSubqueries):
    """
    learning the subqueries: features generation, learning, etc.
    """
    def __init__(self, path, corpus_name):
        super(SubqueriesLearning, self).__init__(path, corpus_name)

        self.features_tmp_root = os.path.join(self.output_root, 'features_tmp')
        if not os.path.exists(self.features_tmp_root):
            os.makedirs(self.features_tmp_root)

        self.subqueries_features_root = os.path.join(self.output_root, 'features')
        if not os.path.exists(self.subqueries_features_root):
            os.makedirs(self.subqueries_features_root)

        self.feature_mapping = {
            1: 'MI',
            2: 'CTF',
            3: 'DF',
            4: 'LOGIDF',
            5: 'MAXTF',
            6: 'MINTF',
            7: 'AVGTF',
            8: 'VARTF'
        }

    def batch_gen_subqueries_features_paras(self, feature_type=0):
        all_paras = []
        for qid in os.listdir(self.subqueries_mapping_root):
            if feature_type != 0:
                output_root = os.path.join(self.subqueries_features_root, self.feature_mapping[feature_type])
                if not os.path.exists(output_root):
                    os.makedirs(output_root)
                feature_outfn = os.path.join(output_root, qid)
                if not os.path.exists(feature_outfn):
                    all_paras.append((self.corpus_path, self.collection_name, qid, feature_type))
        return all_paras

    def gen_subqueries_features(self, qid, feature_type):
        feature_type = int(feature_type)
        if feature_type == 1:
            self.gen_mutual_information(qid)
        elif feature_type == 2:
            self.gen_collection_tf(qid)

    ############## for mutual information ##############
    def run_indri_runquery(self, query_str, runfile_ofn, qid='0', rule=''):
        with open(runfile_ofn, 'w') as f:
            command = ['IndriRunQuery_EX -index=%s -trecFormat=True -count=999999999 -query.number=%s -query.text="%s" -rule=%s' 
                % (os.path.join(self.corpus_path, 'index'), qid, query_str, rule)]
            p = Popen(command, shell=True, stdout=f, stderr=PIPE)
            returncode = p.wait()
            p.communicate()
            if returncode != 0:
                raise NameError("Run Query Error: %s" % (command) )

    def gen_mutual_information(self, qid):
        features_tmp_root = os.path.join(self.features_tmp_root, 'MI')
        if not os.path.exists(features_tmp_root):
            os.makedirs(features_tmp_root)

        features_root = os.path.join(self.subqueries_features_root, 'MI')
        if not os.path.exists(features_root):
            os.makedirs(features_root)

        cs = CollectionStats(self.corpus_path)
        mi_mapping = {}
        withins = [1, 5, 10, 20, 50, 100]
        with open(os.path.join(self.subqueries_mapping_root, qid)) as f:
            subquery_mapping = json.load(f)
        for subquery_id, subquery_str in subquery_mapping.items():
            terms = subquery_str.split()
            if len(terms) == 2: # MI works only for two terms
                terms_stats = {}
                for t in terms:
                    if t not in terms_stats:
                        terms_stats[t] = cs.get_term_stats(t)
                for w in withins:
                    tmp_runfile_fn = os.path.join(features_tmp_root, qid+'_'+subquery_id+'_'+str(w))
                    if not os.path.exists(tmp_runfile_fn):
                        self.run_indri_runquery('#uw%d(%s)' % (w+1, subquery_str), tmp_runfile_fn, rule='method:tf1')
                    ww = 0.0
                    with open(tmp_runfile_fn) as f:
                        for line in f:
                            row = line.split()
                            score = float(row[4])
                            ww += score
                    mi = ww / terms_stats[terms[0]]['total_occur'] if terms_stats[terms[0]]['total_occur'] != 0 else 0.0
                    mi = mi / terms_stats[terms[1]]['total_occur'] if terms_stats[terms[1]]['total_occur'] != 0 else 0.0
                    mi *= cs.get_total_terms()
                    mi = 0 if mi == 0 else np.log(mi)
                    if subquery_str not in mi_mapping:
                        mi_mapping[subquery_str] = {}
                    mi_mapping[subquery_str][w] = mi
        #print json.dumps(mi_mapping, indent=2)
        all_mis = {}
        for subquery_id, subquery_str in subquery_mapping.items():
            terms = subquery_str.split()
            if len(terms) < 2:
                all_mis[subquery_id] = {w:self.get_all_sorts_features([0]) for w in withins}
                continue
            all_mis[subquery_id] = {}
            tmp = {}
            for i in range(len(terms)-1): # including the query itself
                for j in range(i+1, len(terms)):
                    key = terms[i]+' '+terms[j] if terms[i]+' '+terms[j] in mi_mapping else terms[j]+' '+terms[i]
                    for w in mi_mapping[key]:
                        if w not in tmp:
                            tmp[w] = []
                        tmp[w].append(mi_mapping[key][w]) 
            for w in tmp:
                all_mis[subquery_id][w] = self.get_all_sorts_features(tmp[w])
        outfn = os.path.join(features_root, qid)
        with open(outfn, 'wb') as f:
            json.dump(all_mis, f, indent=2)

    ############## for term realted ##############
    def gen_term_related_features(self, qid, feature_formal_name, required_feature):
        features_root = os.path.join(self.subqueries_features_root, feature_formal_name)
        cs = CollectionStats(self.corpus_path)
        with open(os.path.join(self.subqueries_mapping_root, qid)) as f:
            subquery_mapping = json.load(f)
        features = {}
        for subquery_id, subquery_str in subquery_mapping.items():
            terms = subquery_str.split()
            stats = []
            for term in terms:
                print term
                stats.append(cs.get_term_stats(term)[required_feature])
            features[subquery_id] = self.get_all_sorts_features(stats)

        outfn = os.path.join(features_root, qid)
        with open(outfn, 'wb') as f:
            json.dump(features, f, indent=2)


    ############## for collection tf ##############          
    def gen_collection_tf(self, qid):
        self.gen_term_related_features(qid, 'CTF', 'total_occur')
    def gen_df(self, qid):
        self.gen_term_related_features(qid, 'DF', 'df')

    def get_all_sorts_features(self, feature_vec):
        return [np.min(feature_vec), np.max(feature_vec), 
                np.max(feature_vec)-np.min(feature_vec),
                np.max(feature_vec)/np.min(feature_vec) if np.min(feature_vec) != 0 else 0,
                np.mean(feature_vec), np.std(feature_vec), 
                np.sum(feature_vec), 
                0 if np.isnan(scipy.stats.mstats.gmean(feature_vec)) else scipy.stats.mstats.gmean(feature_vec)]

    def sort_subquery_id(self, subquery_id):
        return int(subquery_id.split('-')[0])+float(subquery_id.split('-')[1])/10.0

