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
import xml.etree.ElementTree as ET
from operator import itemgetter
import subprocess
from subprocess import Popen, PIPE
from inspect import currentframe, getframeinfo
import argparse

import numpy as np
import scipy.stats
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from query import Query
from evaluation import Evaluation
from judgment import Judgment
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
            8: 'VARTF',
            9: 'SCS',
            10: 'QLEN',
            11: 'LOGAVGTFIDF',
            12: 'AVGTFCTF',
            13: 'PROXIMITY', # performance score of using proximity query
            14: 'TDC' # TDC looks at the TF relationship in the top docs of ranking list
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
        elif feature_type == 3:
            self.gen_df(qid)
        elif feature_type == 4:
            self.gen_logidf(qid)
        elif feature_type == 5:
            self.gen_maxtf(qid)   
        elif feature_type == 6:
            self.gen_mintf(qid)
        elif feature_type == 7:
            self.gen_avgtf(qid)
        elif feature_type == 8:
            self.gen_vartf(qid)
        elif feature_type == 9:
            self.gen_simple_clarity(qid)
        elif feature_type == 10:
            self.gen_query_len(qid)
        elif feature_type == 11:
            self.gen_logavgtf_idf(qid)
        elif feature_type == 12:
            self.gen_avgtf_cdf(qid)
        elif feature_type == 13:
            self.gen_proximity(qid)
        elif feature_type == 14:
            self.gen_tdc(qid)

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
    def gen_logidf(self, qid):
        self.gen_term_related_features(qid, 'LOGIDF', 'log(idf1)')
    def gen_maxtf(self, qid):
        self.gen_term_related_features(qid, 'MAXTF', 'maxTF')
    def gen_mintf(self, qid):
        self.gen_term_related_features(qid, 'MINTF', 'minTF')
    def gen_avgtf(self, qid):
        self.gen_term_related_features(qid, 'AVGTF', 'avgTF')
    def gen_vartf(self, qid):
        self.gen_term_related_features(qid, 'VARTF', 'varTF')

    def gen_simple_clarity(self, qid):
        features_root = os.path.join(self.subqueries_features_root, 'SCS')
        cs = CollectionStats(self.corpus_path)
        with open(os.path.join(self.subqueries_mapping_root, qid)) as f:
            subquery_mapping = json.load(f)
        features = {}
        for subquery_id, subquery_str in subquery_mapping.items():
            terms = subquery_str.split()
            q_len = len(terms)
            s = 0.0
            for term in terms:
                ctf = cs.get_term_stats(term)['total_occur']
                if ctf == 0:
                    continue
                s += math.log((cs.get_total_terms()*1./q_len/ctf), 2) / q_len
            features[subquery_id] = s

        outfn = os.path.join(features_root, qid)
        with open(outfn, 'wb') as f:
            json.dump(features, f, indent=2)

    def gen_query_len(self, qid):
        features_root = os.path.join(self.subqueries_features_root, 'QLEN')
        with open(os.path.join(self.subqueries_mapping_root, qid)) as f:
            subquery_mapping = json.load(f)
        features = {}
        for subquery_id, subquery_str in subquery_mapping.items():
            features[subquery_id] = len(subquery_str.split())

        outfn = os.path.join(features_root, qid)
        with open(outfn, 'wb') as f:
            json.dump(features, f, indent=2)

    def gen_logavgtf_idf(self, qid):
        features_root = os.path.join(self.subqueries_features_root, self.feature_mapping[11])
        cs = CollectionStats(self.corpus_path)
        with open(os.path.join(self.subqueries_mapping_root, qid)) as f:
            subquery_mapping = json.load(f)
        features = {}
        for subquery_id, subquery_str in subquery_mapping.items():
            terms = subquery_str.split()
            stats = []
            for term in terms:
                avgtf = float(cs.get_term_stats(term)['avgTF'])
                logavgtf = math.log(avgtf) if avgtf-0.0>1e-6 else 0
                logidf = float(cs.get_term_stats(term)['log(idf1)'])
                stats.append(logavgtf*logidf)
            features[subquery_id] = self.get_all_sorts_features(stats)

        outfn = os.path.join(features_root, qid)
        with open(outfn, 'wb') as f:
            json.dump(features, f, indent=2)

    def gen_avgtf_cdf(self, qid):
        features_root = os.path.join(self.subqueries_features_root, self.feature_mapping[12])
        cs = CollectionStats(self.corpus_path)
        with open(os.path.join(self.subqueries_mapping_root, qid)) as f:
            subquery_mapping = json.load(f)
        features = {}
        for subquery_id, subquery_str in subquery_mapping.items():
            terms = subquery_str.split()
            stats = []
            for term in terms:
                avgtf = float(cs.get_term_stats(term)['avgTF'])
                ctf = 1000.*float(cs.get_term_stats(term)['total_occur'])/cs.get_total_terms()
                stats.append(avgtf+ctf)
            features[subquery_id] = self.get_all_sorts_features(stats)

        outfn = os.path.join(features_root, qid)
        with open(outfn, 'wb') as f:
            json.dump(features, f, indent=2)


    def read_runfile_scores(self, fn):
        with open(fn) as f:
            return [float(line.split()[4]) for line in f.readlines()]

    def gen_proximity(self, qid):
        features_root = os.path.join(self.subqueries_features_root, self.feature_mapping[13])
        cs = CollectionStats(self.corpus_path)
        with open(os.path.join(self.subqueries_mapping_root, qid)) as f:
            subquery_mapping = json.load(f)
        features = {}
        type_mapping = {
            1: 'uw',
            2: 'od',
            3: 'uw+od'
        }
        methods = ['dir']
        optimal_lm_performances = Performances(self.corpus_path).load_optimal_performance(methods)[0]
        indri_model_para = 'method:%s,' % optimal_lm_performances[0] + optimal_lm_performances[2]
        for subquery_id, subquery_str in subquery_mapping.items():
            features[subquery_id] = {}
            for _type in sorted(type_mapping):
                name = type_mapping[_type]
                # with open(os.path.join(self.corpus_path, 'subqueries', 'proximity_performances', name, qid+'_'+subquery_id)) as f:
                #     line = f.readline()
                #     row = line.split()
                #     try:
                #         p = float(row[-1])
                #     except:
                #         p = 0.0
                #     features[subquery_id].append(p)
                orig_runfile_fn = os.path.join(self.subqueries_runfiles_root, qid+'_'+subquery_id+'_'+indri_model_para)
                proximity_runfile_fn = os.path.join(self.corpus_path, 'subqueries', 'proximity_runfiles', name, qid+'_'+subquery_id)
                orig_ranking_scores = self.read_runfile_scores(orig_runfile_fn)
                prox_ranking_scores = self.read_runfile_scores(proximity_runfile_fn)
                if orig_ranking_scores:
                    orig_features = self.get_all_sorts_features(orig_ranking_scores)
                else:
                    orig_features = [0, 0, 0, 0, 0, 0, 0, 0]
                    # orig_features = [0, 0, 0]
                if prox_ranking_scores:
                    prox_features = self.get_all_sorts_features(prox_ranking_scores)
                else:
                    prox_features = [0, 0, 0, 0, 0, 0, 0, 0]
                    # prox_features = [0, 0, 0]
                diff_features = np.array(prox_features) - np.array(orig_features)
                # correlation between proximity ranking list and orig ranking list
                rankinglist_cutoff = 50
                if len(orig_ranking_scores) < rankinglist_cutoff or len(prox_ranking_scores) < rankinglist_cutoff:
                    rankinglist_cutoff = min(len(orig_ranking_scores), len(prox_ranking_scores))
                tau, p_tau = scipy.stats.kendalltau(prox_ranking_scores[: rankinglist_cutoff], orig_ranking_scores[: rankinglist_cutoff])
                pea, p_r = scipy.stats.pearsonr(prox_ranking_scores[: rankinglist_cutoff], orig_ranking_scores[: rankinglist_cutoff])
                tau = tau if not np.isnan(tau) else 0
                pea = pea if not np.isnan(pea) else 0
                features[subquery_id][name] = []
                features[subquery_id][name].extend(orig_features)
                features[subquery_id][name].extend(prox_features)
                features[subquery_id][name].extend(diff_features)
                features[subquery_id][name].append(tau)
                features[subquery_id][name].append(pea)

        outfn = os.path.join(features_root, qid)
        with open(outfn, 'wb') as f:
            json.dump(features, f, indent=2)

    def gen_tdc(self, qid, _type=2):
        features_root = os.path.join(self.subqueries_features_root, 'TDC')
        if not os.path.exists(features_root):
            os.makedirs(features_root)

        cs = CollectionStats(self.corpus_path)
        all_features = {}
        # withins = [1, 5, 10, 20, 50, 100]
        withins = [50]
        features_wpara = [[] for ele in withins]
        methods = ['okapi']
        optimal_lm_performances = Performances(self.corpus_path).load_optimal_performance(methods)[0]
        indri_model_para = 'method:%s,' % optimal_lm_performances[0] + optimal_lm_performances[2]
        model_para = float(optimal_lm_performances[2].split(':')[1])
        with open(os.path.join(self.subqueries_mapping_root, qid)) as f:
            subquery_mapping = json.load(f)

        for subquery_id, subquery_str in subquery_mapping.items():
            orig_runfile_fn = os.path.join(self.subqueries_runfiles_root, qid+'_'+subquery_id+'_'+indri_model_para)
            with open(orig_runfile_fn) as f:
                line_idx = 0
                for line in f:
                    line = line.strip()
                    if line:
                        row = line.split()
                        tf_details = row[1]
                        terms = [ele.split('-')[0] for ele in tf_details.split(',')]
                        tfs = [float(ele.split('-')[1]) for ele in tf_details.split(',')]
                        dl = float(row[-1].split(',')[0].split(':')[1])
                        if _type == 1: # simple TF
                            scores = tfs
                        elif _type == 2: # BM25
                            scores = [tf*cs.get_term_logidf1(terms[i])*2.2/(tf+1.2*(1-model_para+model_para*dl/cs.get_avdl())) for i, tf in enumerate(tfs)]
                        tf_features = self.get_all_sorts_features(scores)
                        for i, w in enumerate(withins):
                            if line_idx < w:
                                features_wpara[i].append(tf_features)
                    line_idx += 1
                    if line_idx >= 100:
                        break
            all_features[subquery_id] = {}
            for i, w in enumerate(withins):
                all_features[subquery_id][w] = []
                for column in np.array(features_wpara[i]).T:
                    all_features[subquery_id][w].extend(self.get_all_sorts_features(column))

        outfn = os.path.join(features_root, qid)
        with open(outfn, 'wb') as f:
            json.dump(all_features, f, indent=2)


    def get_all_sorts_features(self, feature_vec):
        return [np.min(feature_vec), np.max(feature_vec), 
                np.max(feature_vec)-np.min(feature_vec),
                np.max(feature_vec)/np.min(feature_vec) if np.min(feature_vec) != 0 else 0,
                np.mean(feature_vec), np.std(feature_vec), 
                np.sum(feature_vec), 
                0 if np.isnan(scipy.stats.mstats.gmean(feature_vec)) else scipy.stats.mstats.gmean(feature_vec)]
        # return [np.mean(feature_vec), np.std(feature_vec), 
        #         0 if np.isnan(scipy.stats.mstats.gmean(feature_vec)) else scipy.stats.mstats.gmean(feature_vec)]



    def sort_qid(self, qid):
        return int(qid)
    def sort_subquery_id(self, subquery_id):
        return int(subquery_id.split('-')[0])+float(subquery_id.split('-')[1])/10.0


    def get_feature_mapping(self):
        mapping = {}
        idx = 1
        features = ['min', 'max', 'max-min', 'max/min', 'mean', 'std', 'sum', 'gmean']
        # features = ['mean', 'std', 'gmean']
        for feature_idx, feature_name in self.feature_mapping.items():
            if feature_idx == 1: # mutual information
                #withins = [1, 5, 10, 20, 50, 100]
                withins = [50]
                for w in withins:
                    for fa in features:
                        mapping[idx] = feature_name+str(w)+'('+fa+')'
                        idx += 1
            elif feature_idx == 9 or feature_idx == 10: # query length and Clarity
                mapping[idx] = feature_name
                idx += 1
            elif feature_idx == 13: # proximity query features
                withins = ['uw', 'od', 'uw+od']
                feature_types = ['orig', 'prox', 'diff']
                for w in withins:
                    for t in feature_types:
                        for fa in features:
                            mapping[idx] = feature_name+str(w)+'('+t+'-'+fa+')'
                            idx += 1
                    mapping[idx] = feature_name+str(w)+'(ktau)'
                    idx += 1
                    mapping[idx] = feature_name+str(w)+'(pear)'
                    idx += 1
            elif feature_idx == 14: # TDC
                withins = [50]
                for w in withins:
                    for fa in features: # on a single doc
                        for fb in features: # on the column
                            mapping[idx] = feature_name+str(w)+'('+fa+'-'+fb+')'
                            idx += 1
            else:
                for fa in features:
                    mapping[idx] = feature_name+'('+fa+')'
                    idx += 1
        return mapping

    def get_all_features(self, query_length=0):
        q = Query(self.corpus_path)
        if query_length == 0:
            queries = q.get_queries()
        else:
            queries = q.get_queries_of_length(query_length)
        queries = {ele['num']:ele['title'] for ele in queries}

        all_features = {}
        for qid in os.listdir(self.subqueries_mapping_root):
            if qid not in queries:
                continue
            all_features[qid] = {}
            for feature_idx, feature_name in self.feature_mapping.items():
                features_root = os.path.join(self.subqueries_features_root, feature_name)
                with open(os.path.join(features_root, qid)) as f:
                    qid_features = json.load(f)
                for subquery_id in sorted(qid_features, key=self.sort_subquery_id):
                    if subquery_id not in all_features[qid]:
                        all_features[qid][subquery_id] = []
                    if feature_idx == 1: # mutual information
                        # withins = [1, 5, 10, 20, 50, 100]
                        withins = [50]
                        for w in withins:
                            str_w = str(w)
                            all_features[qid][subquery_id].extend(qid_features[subquery_id][str_w])
                    elif feature_idx == 14: # TDC
                        withins = [50]
                        for w in withins:
                            str_w = str(w)
                            all_features[qid][subquery_id].extend(qid_features[subquery_id][str_w])
                    elif feature_idx == 9 or feature_idx == 10: # query length and Clarity
                        all_features[qid][subquery_id].append(qid_features[subquery_id])
                    elif feature_idx == 13: # proximity query
                        withins = ['uw', 'od', 'uw+od']
                        for w in withins:
                            all_features[qid][subquery_id].extend(qid_features[subquery_id][w])
                    else:
                        all_features[qid][subquery_id].extend(qid_features[subquery_id])
        return all_features

    def get_all_performances(self, model='okapi'):
        results = {}
        for fn in os.listdir(self.subqueries_performance_root):
            fn_split = fn.split('_')
            qid = fn_split[0]
            subquery_id = fn_split[1]
            model_para = fn_split[2]
            with open(os.path.join(self.subqueries_mapping_root, qid)) as f:
                subquery_mapping = json.load(f)
            try:
                with open(os.path.join(self.subqueries_performance_root, fn)) as f:
                    first_line = f.readline()
                    ap = float(first_line.split()[-1])
            except:
                continue
            if qid not in results:
                results[qid] = {}
            if model in model_para:
                results[qid][subquery_id] = ap

        return results

    def output_correlation_features(self, query_len=0, corr_type=1):
        """
        corr_type: 1-kendallstau, 2-pearsonr, 3-ken_n_pea
        output the correlation between features and the ranking of subqueries.
        The output can be used to reduce the dimension of the feature space
        """
        if corr_type == 1:
            corr_type_str = 'kendallstau'
        elif corr_type == 2:
            corr_type_str = 'pearsonr'
        elif corr_type == 3:
            corr_type_str = 'ken_n_pea'
        output_root = os.path.join(self.subqueries_features_root, corr_type_str)
        if not os.path.exists(output_root):
            os.makedirs(output_root)
        output_fn = os.path.join(output_root, str(query_len))
        feature_mapping = self.get_feature_mapping()
        all_performances = self.get_all_performances()
        all_features = self.get_all_features(query_len)
        all_features_matrix = []
        kendallstau = {}
        pearsonr = {}
        ken_n_pea = {}
        for qid in sorted(all_features):
            for subquery_id in sorted(all_features[qid]):
                all_features_matrix.append(all_features[qid][subquery_id])
            this_features = np.array([all_features[qid][subquery_id] for subquery_id in sorted(all_features[qid])])
            if this_features.shape[0] == 0:
                continue
            this_perfm = [float(all_performances[qid][subquery_id]) if qid in all_performances and subquery_id in all_performances[qid] else 0.0 for subquery_id in sorted(all_features[qid])]
            for col in range(this_features.shape[1]):
                tau, p_tau = scipy.stats.kendalltau(this_features[:, col], this_perfm)
                pea, p_r = scipy.stats.pearsonr(this_features[:, col], this_perfm)
                if col+1 not in kendallstau:
                    kendallstau[col+1] = []
                if col+1 not in pearsonr:
                    pearsonr[col+1] = []
                kendallstau[col+1].append(tau if not np.isnan(tau) else 0)
                pearsonr[col+1].append(pea if not np.isnan(pea) else 0)
        if corr_type == 1:
            klist = [(col, np.mean(kendallstau[col])) for col in kendallstau]
        elif corr_type == 2:
            klist = [(col, np.mean(pearsonr[col])) for col in pearsonr]
        klist.sort(key=itemgetter(1), reverse=True)
        top_features = [ele[0] for ele in klist[:10]]
        print [[ele, feature_mapping[ele], klist[i][1]] for i, ele in enumerate(top_features)]

        normalized = normalize(all_features_matrix, axis=0) # normalize each feature
        idx = 0
        with open(output_fn+'.ap', 'wb') as f:
            for qid in sorted(all_features, key=self.sort_qid):
                for subquery_id in sorted(all_features[qid], key=self.sort_subquery_id):
                    ### sample training: "3 qid:1 1:1 2:1 3:0 4:0.2 5:0 # 1A"
                    if qid in all_performances and subquery_id in all_performances[qid]:
                        f.write('%s qid:%s %s # %s\n' % (str(all_performances[qid][subquery_id]), qid, 
                            ' '.join(['%d:%f' % (i, normalized[idx][i-1]) for i in range(1, len(normalized[idx])+1) if i in top_features]), 
                            subquery_id))
                    idx += 1
        idx = 0
        with open(output_fn+'.int', 'wb') as f:
            for qid in sorted(all_features, key=self.sort_qid):
                tmp_label = []
                for subquery_id in sorted(all_features[qid], key=self.sort_subquery_id):
                    ### sample training: "3 qid:1 1:1 2:1 3:0 4:0.2 5:0 # 1A"
                    if qid in all_performances and subquery_id in all_performances[qid]:
                        tmp_label.append(all_performances[qid][subquery_id])
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
                tmp_label_idx = 0
                for subquery_id in sorted(all_features[qid], key=self.sort_subquery_id):
                    ### sample training: "3 qid:1 1:1 2:1 3:0 4:0.2 5:0 # 1A"
                    if qid in all_performances and subquery_id in all_performances[qid]:
                        f.write('%d qid:%s %s # %s\n' % (tmp_label[tmp_label_idx], qid, 
                            ' '.join(['%d:%f' % (i, normalized[idx][i-1]) for i in range(1, len(normalized[idx])+1) if i in top_features]), 
                            subquery_id))
                        tmp_label_idx += 1
                    idx += 1

    def output_features_selected(self, query_len=0):
        """
        output the selected features.
        The selected features are carefully selected manually.
        """
        output_root = os.path.join(self.subqueries_features_root, 'selected')
        if not os.path.exists(output_root):
            os.makedirs(output_root)
        output_fn = os.path.join(output_root, str(query_len))
        feature_mapping = self.get_feature_mapping()
        all_performances = self.get_all_performances()
        all_features = self.get_all_features(query_len)
        print all_features
        exit()
        all_features_matrix = []
        selected = {}
        for qid in sorted(all_features):
            for subquery_id in sorted(all_features[qid]):
                all_features_matrix.append(all_features[qid][subquery_id])
            this_features = np.array([all_features[qid][subquery_id] for subquery_id in sorted(all_features[qid])])
            if this_features.shape[0] == 0:
                continue
            this_perfm = [float(all_performances[qid][subquery_id]) if qid in all_performances and subquery_id in all_performances[qid] else 0.0 for subquery_id in sorted(all_features[qid])]
            for col in range(this_features.shape[1]):
                tau, p_value = scipy.stats.kendalltau(this_features[:, col], this_perfm)
                if col+1 not in kendallstau:
                    kendallstau[col+1] = []
                kendallstau[col+1].append(tau if not np.isnan(tau) else 0)
        klist = [(col, np.mean(kendallstau[col])) for col in kendallstau]
        klist.sort(key=itemgetter(1), reverse=True)
        top_features = [ele[0] for ele in klist[:10]]
        print top_features

        normalized = normalize(all_features_matrix, axis=0) # normalize each feature
        idx = 0
        with open(output_fn, 'wb') as f: 
            for qid in sorted(all_features, key=self.sort_qid):
                for subquery_id in sorted(all_features[qid], key=self.sort_subquery_id):
                    ### sample training: "3 qid:1 1:1 2:1 3:0 4:0.2 5:0 # 1A"
                    if qid in all_performances and subquery_id in all_performances[qid]:
                        f.write('%s qid:%s %s # %s\n' % (str(all_performances[qid][subquery_id]), qid, 
                            ' '.join(['%d:%f' % (i, normalized[idx][i-1] if i in top_features else 0) for i in range(1, len(normalized[idx])+1)]), 
                            subquery_id))
                    idx += 1

    @staticmethod
    def output_correlation_features_all_collection(collection_paths_n_names, query_length=0, corr_type=1):
        if corr_type == 1:
            corr_type_str = 'kendallstau'
        elif corr_type == 2:
            corr_type_str = 'pearsonr'
        elif corr_type == 3:
            corr_type_str = 'ken_n_pea'
        all_features = {}
        feature_mapping = {}
        for ele in collection_paths_n_names:
            collection_path = ele[0]
            collection_name = ele[1]
            q = Query(collection_path)
            if query_length == 0:
                queries = q.get_queries()
            else:
                queries = q.get_queries_of_length(query_length)
            queries = {ele['num']:ele['title'] for ele in queries}
            with open(os.path.join(collection_path, 'subqueries', 'features', corr_type_str, str(query_length))) as f:
                r = csv.reader(f)
                for row in r:
                    feature_id = row[0]
                    feature_name = row[1]
                    feature_mapping[feature_id] = feature_name
                    feature_score = float(row[2]) * len(queries)
                    if feature_id not in all_features:
                        all_features[feature_id] = 0.0
                    all_features[feature_id] += feature_score
        sorted_f = sorted(all_features.items(), key=itemgetter(1), reverse=True)
        for ele in sorted_f[:10]:
            print ele[0], feature_mapping[ele[0]], ele[1]

    @staticmethod
    def load_optimal_ground_truth(collection_path, qids):
        optimal_ground_truth = {}
        using_all_terms = {}
        second_optimal = {}
        root = os.path.join(collection_path, 'subqueries', 'collected_results')
        for qid in qids:
            qid_performances = []
            with open(os.path.join(root, qid)) as f:
                csvr = csv.reader(f)
                ap_all_terms = 0.0
                for row in csvr:
                    subquery_id = row[0]
                    subquery = row[1]
                    model_para = row[2]
                    ap = float(row[3])
                    if 'okapi' in model_para:
                        qid_performances.append((subquery_id, ap))
                        ap_all_terms = ap
            qid_performances.sort(key=itemgetter(1), reverse=True)
            optimal_ground_truth[qid] = qid_performances[0][1]
            second_optimal[qid] = qid_performances[1][1]
            using_all_terms[qid] = ap_all_terms
        return optimal_ground_truth, using_all_terms, second_optimal

    def output_collection_features(self, query_len=0):
        """
        output the collection level features to output
        so that it can be fed to SVMRank
        for each qid the training instances are the subqueries.
        """
        feature_mapping = self.get_feature_mapping()
        with open(os.path.join(self.subqueries_features_root, 'mapping'), 'wb') as f: 
            json.dump(feature_mapping, f, indent=2)
        output_root = os.path.join(self.subqueries_features_root, 'final')
        if not os.path.exists(output_root):
            os.makedirs(output_root)
        all_features = self.get_all_features(query_len)
        all_features_matrix = []
        for qid in sorted(all_features):
            for subquery_id in sorted(all_features[qid]):
                all_features_matrix.append(all_features[qid][subquery_id])
        normalized = normalize(all_features_matrix, axis=0) # normalize each feature
        all_performances = self.get_all_performances()
        idx = 0
        with open(os.path.join(output_root, str(query_len)+'.ap'), 'wb') as f: 
            for qid in sorted(all_features, key=self.sort_qid):
                for subquery_id in sorted(all_features[qid], key=self.sort_subquery_id):
                    ### sample training: "3 qid:1 1:1 2:1 3:0 4:0.2 5:0 # 1A"
                    if qid in all_performances and subquery_id in all_performances[qid]:
                        f.write('%s qid:%s %s # %s\n' % (str(all_performances[qid][subquery_id]), qid, 
                            ' '.join(['%d:%f' % (i, normalized[idx][i-1]) for i in range(1, len(normalized[idx])+1)]), 
                            subquery_id))
                    idx += 1
        idx = 0
        with open(os.path.join(output_root, str(query_len)+'.int'), 'wb') as f:
            for qid in sorted(all_features, key=self.sort_qid):
                tmp_label = []
                for subquery_id in sorted(all_features[qid], key=self.sort_subquery_id):
                    ### sample training: "3 qid:1 1:1 2:1 3:0 4:0.2 5:0 # 1A"
                    if qid in all_performances and subquery_id in all_performances[qid]:
                        tmp_label.append(all_performances[qid][subquery_id])
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
                tmp_label_idx = 0
                for subquery_id in sorted(all_features[qid], key=self.sort_subquery_id):
                    ### sample training: "3 qid:1 1:1 2:1 3:0 4:0.2 5:0 # 1A"
                    if qid in all_performances and subquery_id in all_performances[qid]:
                        f.write('%d qid:%s %s # %s\n' % (tmp_label[tmp_label_idx], qid, 
                            ' '.join(['%d:%f' % (i, normalized[idx][i-1]) for i in range(1, len(normalized[idx])+1)]), 
                            subquery_id))
                        tmp_label_idx += 1
                    idx += 1

    def batch_gen_learning_to_rank_paras(self, feature_type=1, method=1, label_type='int'):
        if feature_type == 2:
            folder = 'kendallstau'
        elif feature_type == 3:
            folder = 'pearsonr'
        else:
            folder = 'final'

        if method == 1:
            method_folder = 'svm_rank'
        elif method == 2:
            method_folder = 'lambdamart'
        paras = []
        model_root = os.path.join(self.output_root, method_folder, folder, 'models')
        if not os.path.exists(model_root):
            os.makedirs(model_root)
        predict_root = os.path.join(self.output_root, method_folder, folder, 'predict')
        if not os.path.exists(predict_root):
            os.makedirs(predict_root)
        for fn in os.listdir(os.path.join(self.subqueries_features_root, folder)):
            fn_splits = fn.split('.')
            if len(fn_splits) != 2 or fn_splits[1] != label_type:
                continue
            if method == 1:
                for c in range(-5, 5):
                    if not os.path.exists(os.path.join(model_root, fn+'_'+str(10**c))):
                        paras.append((self.corpus_path, self.collection_name, folder, fn, method, c))
            elif method == 2:
                for leaf in range(2, 10):
                    if not os.path.exists(os.path.join(model_root, fn+'_'+str(leaf))):
                        paras.append((self.corpus_path, self.collection_name, folder, fn, method, leaf))
        return paras

    def learning_to_rank_wrapper(self, folder, feature_fn, method, method_para):
        if method == 1:
            method_folder = 'svm_rank'
        elif method == 2:
            method_folder = 'lambdamart'
        model_root = os.path.join(self.output_root, method_folder, folder, 'models')
        if method == 1:
            c = int(method_para)
            command = ['svm_rank_learn', '-c', str(10**c), 
                os.path.join(self.subqueries_features_root, folder, feature_fn), 
                os.path.join(model_root, feature_fn+'_'+str(10**c))]
            subprocess.call(command)
        elif method == 2:
            leaf = int(method_para)
            command = 'java -jar -Xmx2g ~/Downloads/RankLib-2.8.jar -train %s -ranker 6 -leaf %d -save %s' % ( 
                os.path.join(self.subqueries_features_root, folder, feature_fn), 
                leaf,
                os.path.join(model_root, feature_fn+'_'+str(leaf)))
            p = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
            returncode = p.wait()
            out, error = p.communicate()
            if returncode != 0:
                raise NameError("Run Query Error: %s" % (command) )
            print out

    def read_data_from_feature_file(self, fn):
        data = []
        labels = []
        with open(fn) as f:
            for line in f:
                line = line.strip()
                if line:
                    row = line.split()
                    row_data = [float(ele.split(':')[-1]) for ele in row[2:-2]] # do not include qid
                    data.append(row_data)
                    qid = row[1].split(':')[1]
                    label = float(row[0])
                    labels.append(label)
        return data, labels

    def read_lambdamart_model_file(self, fn):
        with open(fn) as f:
            root = ET.fromstring('\n'.join(f.readlines()[6:]))
        all_features = []
        for tree in root.findall('tree'):
            all_features.append(int(tree.find('split').find('feature').text))
        return all_features

    def evaluate_learning_to_rank_model(self, feature_type=1, method=1):
        if feature_type == 2:
            folder = 'kendallstau'
        elif feature_type == 3:
            folder = 'pearsonr'
        else:
            folder = 'final'
        if method == 1:
            method_folder = 'svm_rank'
        elif method == 2:
            method_folder = 'lambdamart'
        model_root = os.path.join(self.output_root, method_folder, folder, 'models')
        predict_root = os.path.join(self.output_root, method_folder, folder, 'predict')
        all_models = {}
        error_rate_fn = os.path.join(self.output_root, method_folder, folder, 'err_rate')
        error_rates = {}
        for fn in os.listdir(model_root):
            predict_output_fn = os.path.join(predict_root, fn)
            if os.path.exists(predict_output_fn) and os.path.exists(error_rate_fn):
                continue
            feature_fn = fn.split('_')[0]
            label_type = feature_fn.split('.')[1]
            para = fn.split('_')[1]
            if method == 1:
                command = ['svm_rank_classify %s %s %s' 
                    % (os.path.join(self.subqueries_features_root, folder, feature_fn), 
                        os.path.join(model_root, fn), 
                        os.path.join(predict_root, fn))]
            elif method == 2:
                command = ['java -jar -Xmx2g ~/Downloads/RankLib-2.8.jar -train %s -ranker 6 -leaf %s' 
                    % (os.path.join(self.subqueries_features_root, folder, feature_fn), para)]
            p = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
            returncode = p.wait()
            out, error = p.communicate()
            if returncode != 0:
                raise NameError("Run Query Error: %s" % (command) )
            query_length = int(feature_fn.split('.')[0])
            if method == 1:
                err_rate = float(out.split('\n')[-2].split(':')[1])
            elif method == 2:
                err_rate = 1-float(out.split('\n')[-3].split(':')[1])
                command = ['java -jar -Xmx2g ~/Downloads/RankLib-2.8.jar -load %s -rank %s -score %s'
                    % (os.path.join(model_root, fn), 
                    os.path.join(self.subqueries_features_root, folder, feature_fn),
                    os.path.join(predict_root, fn))]
                subprocess.call(command, shell=True)
            if query_length not in error_rates:
                error_rates[query_length] = {}
            if label_type not in error_rates[query_length]:
                error_rates[query_length][label_type] = {}
            error_rates[query_length][label_type][para] = err_rate
            if query_length not in all_models:
                all_models[query_length] = {}
            if label_type not in all_models[query_length]:
                all_models[query_length][label_type] = []
            all_models[query_length][label_type].append((para, err_rate))
        if error_rates:
            with open(error_rate_fn, 'wb') as f:
                json.dump(error_rates, f, indent=2)
        if not all_models:
            with open(error_rate_fn) as f:
                error_rates = json.load(f)
            for query_length in error_rates:
                if query_length not in all_models:
                    all_models[query_length] = {}
                for label_type in error_rates[query_length]:
                    if label_type not in all_models[query_length]:
                        all_models[query_length][label_type] = []
                    for para in error_rates[query_length][label_type]:
                        all_models[query_length][label_type].append((para, error_rates[query_length][label_type][para]))
        feature_mapping = self.get_feature_mapping()
        predict_optimal_subquery_len_dist = {}
        with open(os.path.join(self.final_output_root, self.collection_name+'-%s_subquery_dist-%s.md' % (method_folder, folder)), 'wb') as ssdf:
            ssdf.write('### %s\n' % (self.collection_name))
            ssdf.write('| query len | label_type | using all terms | optimal (ground truth) | optimal |\n')
            ssdf.write('|--------|--------|--------|--------|--------|\n')
            for query_length in sorted(all_models):
                predict_optimal_subquery_len_dist[query_length] = {}
                # first sort based on err_rate
                for label_type in all_models[query_length]:
                    all_models[query_length][label_type].sort(key=itemgetter(1))

                    # model prediction performance related
                    predict_optimal_subquery_len_dist[query_length][label_type] = {}
                    predict_optimal_performance = {}
                    existing_performance = {}
                    optimal_ground_truth = 0.0
                    optimal_model_predict = 0.0
                    performance_using_all_terms = 0.0
                    para = all_models[query_length][label_type][0][0]
                    feature_fn = os.path.join(self.subqueries_features_root, folder, str(query_length)+'.'+label_type)
                    predict_fn = os.path.join(predict_root, str(query_length)+'.'+label_type+'_'+para)
                    with open(predict_fn) as f:
                        if method == 1:
                            predict_res = [float(line.strip()) for line in f.readlines()]
                        elif method == 2:
                            predict_res = [float(line.split()[-1].strip()) for line in f.readlines()]
                    with open(feature_fn) as f:
                        idx = 0
                        for line in f:
                            line = line.strip()
                            row = line.split()
                            qid = row[1].split(':')[1]
                            subquery_id = row[-1]
                            if qid not in predict_optimal_performance:
                                predict_optimal_performance[qid] = []
                                # read the performances of okapi and dirichlet
                                existing_performance[qid] = {}
                                qid_performances = []
                                with open(os.path.join(self.collected_results_root, qid)) as f:
                                    csvr = csv.reader(f)
                                    for row in csvr:
                                        subquery_id = row[0]
                                        subquery = row[1]
                                        model_para = row[2]
                                        ap = float(row[3])
                                        if 'okapi' in model_para:
                                            qid_performances.append((subquery_id, ap))
                                            existing_performance[qid][subquery_id] = ap
                                performance_using_all_terms += qid_performances[-1][1]
                                qid_performances.sort(key=itemgetter(1), reverse=True)
                                optimal_ground_truth += qid_performances[0][1]
                            predict_optimal_performance[qid].append((subquery_id, predict_res[idx], existing_performance[qid][subquery_id]))
                            idx += 1
                    for qid in predict_optimal_performance:
                        predict_optimal_performance[qid].sort(key=itemgetter(1), reverse=True)
                        optimal_model_predict += predict_optimal_performance[qid][0][2]
                        subquery_len = int(predict_optimal_performance[qid][0][0].split('-')[0])
                        if subquery_len not in predict_optimal_subquery_len_dist[query_length][label_type]:
                            predict_optimal_subquery_len_dist[query_length][label_type][subquery_len] = 0
                        predict_optimal_subquery_len_dist[query_length][label_type][subquery_len] += 1

                    query_cnt = len(predict_optimal_performance)
                    ssdf.write('| %s | %s | %.4f | %.4f | %.4f |\n' 
                        % ( query_length, label_type,
                            performance_using_all_terms/query_cnt, 
                            optimal_ground_truth/query_cnt, 
                            optimal_model_predict/query_cnt))

                    # feature ranking related
                    model_fn = str(query_length)+'.'+str(label_type)+'_'+str(all_models[query_length][label_type][0][0])
                    if method == 1:
                        with open(os.path.join(model_root, model_fn)) as f:
                            model = f.readlines()[-1]
                        feature_weights = [(int(ele.split(':')[0]), float(ele.split(':')[1])) for ele in model.split()[1:-1]]
                    elif method == 2:
                        # train_data, train_label = self.read_data_from_feature_file(feature_fn)
                        # clf = GradientBoostingRegressor(n_estimators=1000, max_depth=1, random_state=0).fit(train_data, train_label)
                        # print clf.feature_importances_
                        all_features = self.read_lambdamart_model_file(os.path.join(model_root, model_fn))
                        features_dict = {}
                        for feature in all_features:
                            if feature not in features_dict:
                                features_dict[feature] = 0
                            features_dict[feature] += 1
                        feature_weights = [(k,v) for k,v in features_dict.items()]
                    feature_weights.sort(key=itemgetter(1, 0), reverse=True)
                    output_root = os.path.join(self.output_root, method_folder, folder, 'featurerank')
                    if not os.path.exists(output_root):
                        os.makedirs(output_root)
                    with open(os.path.join(output_root, str(query_length)), 'wb') as f:
                        for ele in feature_weights:
                            f.write('%s: %f\n' % (feature_mapping[ele[0]], ele[1]))

            ssdf.write('\n#### predict subquery length distribution\n')
            ssdf.write('| | | | | |\n')
            ssdf.write('|--------|--------|--------|--------|--------|\n')
            for query_len in predict_optimal_subquery_len_dist:
                for label_type in predict_optimal_subquery_len_dist[query_len]:
                    ssdf.write('| %s | %s |' % (query_len, label_type))
                    for subquery_len in predict_optimal_subquery_len_dist[query_len][label_type]:
                        ssdf.write(' %d:%d |' % (subquery_len, predict_optimal_subquery_len_dist[query_len][label_type][subquery_len]))
                    ssdf.write('\n')

    @staticmethod
    def write_combined_feature_fn(results_root, l, ofn, query_length=2, reorder_qid=False, _type='int'):
        trainging_fn = os.path.join(results_root, 'train_%d' % query_length)
        if os.path.exists(ofn):
            os.remove(ofn)
        with open(ofn, 'ab') as f:
            qid_idx = 1
            qid_lines = {}
            for ele in l:
                collection_path = ele[0]
                collection_name = ele[1]
                feature_fn = os.path.join(collection_path, 'subqueries', 'features', 'final', str(query_length)+'.'+_type)
                with open(feature_fn) as ff:
                    if not reorder_qid:
                        f.write(ff.read())
                    else:
                        for line in ff:
                            line = line.strip()
                            row = line.split()
                            qid = int(row[1].split(':')[1])
                            if qid not in qid_lines:
                                qid_lines[qid] = []
                            qid_lines[qid].append(line)
            if reorder_qid:
                for qid in qid_lines:
                    for line in qid_lines[qid]:
                        row = line.split()
                        row[1] = 'qid:%d' % qid_idx
                        f.write('%s\n' % ' '.join(row))
                    qid_idx += 1

    @staticmethod
    def evaluate_learning_to_rank_cross_testing(all_data, method=1, label_type='int'):
        if method == 1:
            method_folder = 'svm_rank'
        elif method == 2:
            method_folder = 'lambdamart'
        data_mapping = {d[1]:d[0] for d in all_data}
        results_root = os.path.join('../all_results', 'subqueries', 'cross_training', label_type, method_folder)
        all_predict_data = {}
        for fn in os.listdir(results_root):
            m = re.search(r'^predict_(.*?)_(.*?)_(.*)$', fn)
            if m:
                collection_name = m.group(1)
                query_length = int(m.group(2))
                para = m.group(3)
                if query_length not in all_predict_data:
                    all_predict_data[query_length] = {}
                if para not in all_predict_data[query_length]:
                    all_predict_data[query_length][para] = []
                all_predict_data[query_length][para].append(collection_name)

        all_performances = {}
        for query_length in all_predict_data:
            all_performances[query_length] = []
            for para in all_predict_data[query_length]:
                if len(all_predict_data[query_length][para]) != len(all_data):
                    print 'query length: %d and para: %s does not have enough data ... %d/%d' \
                        % (query_length, para, len(all_predict_data[query_length][para]), len(all_data))
                    continue
                #svm_predict_optimal_subquery_len_dist[query_length] = {}
                existing_performance = {}
                collection_predict_performance = {}
                optimal_ground_truth = 0.0
                optimal_svm_predict = 0.0
                performance_using_all_terms = 0.0
                for collection_name in all_predict_data[query_length][para]: 
                    predict_optimal_performance = {}
                    feature_fn = os.path.join(results_root, 'test_%s_%d' % (collection_name, query_length))
                    predict_fn = os.path.join(results_root, 'predict_%s_%d_%s' % (collection_name, query_length, para))
                    with open(predict_fn) as f:
                        if method == 1:
                            predict_res = [float(line.strip()) for line in f.readlines()]
                        elif method == 2:
                            predict_res = [float(line.strip().split()[-1]) for line in f.readlines()]
                    with open(feature_fn) as f:
                        idx = 0
                        for line in f:
                            line = line.strip()
                            row = line.split()
                            qid = row[1].split(':')[1]
                            subquery_id = row[-1]
                            if qid not in predict_optimal_performance:
                                predict_optimal_performance[qid] = []
                                # read the performances of okapi and dirichlet
                                existing_performance[qid] = {}
                                qid_performances = []
                                with open(os.path.join(data_mapping[collection_name], 'subqueries', 'collected_results', qid)) as f:
                                    csvr = csv.reader(f)
                                    for row in csvr:
                                        subquery_id = row[0]
                                        subquery = row[1]
                                        model_para = row[2]
                                        ap = float(row[3])
                                        if 'okapi' in model_para:
                                            qid_performances.append((subquery_id, ap))
                                            existing_performance[qid][subquery_id] = ap
                                performance_using_all_terms += qid_performances[-1][1]
                                qid_performances.sort(key=itemgetter(1), reverse=True)
                                optimal_ground_truth += qid_performances[0][1]
                            predict_optimal_performance[qid].append((subquery_id, predict_res[idx], existing_performance[qid][subquery_id]))
                            idx += 1
                    collection_predict = 0.0
                    for qid in predict_optimal_performance:
                        predict_optimal_performance[qid].sort(key=itemgetter(1), reverse=True)
                        collection_predict += predict_optimal_performance[qid][0][2]
                        optimal_svm_predict += collection_predict
                        subquery_len = int(predict_optimal_performance[qid][0][0].split('-')[0])
                        # if subquery_len not in svm_predict_optimal_subquery_len_dist[query_length]:
                        #     svm_predict_optimal_subquery_len_dist[query_length][subquery_len] = 0
                        # svm_predict_optimal_subquery_len_dist[query_length][subquery_len] += 1 
                    collection_predict_performance[collection_name] = collection_predict / len(predict_optimal_performance)        
                all_performances[query_length].append((para, optimal_svm_predict, collection_predict_performance))
            all_performances[query_length].sort(key=itemgetter(1), reverse=True)

        print 'Method: %s' % method_folder
        for query_length in all_performances:
            print query_length, json.dumps(all_performances[query_length][0], indent=2)

    @staticmethod
    def cross_testing_learning_to_rank_model(train, test, query_length=2, method=1, label_type='int'):
        """
        train and test are list of (collection_path, collection_name)
        # label_type: int (use integer as labels) or ap (use ap value floating numbers as labels)
        method: 1: svm_rank, 2: lambdamart
        """
        test_collection = test[0][1]
        if method == 1:
            method_folder = 'svm_rank'
        elif method == 2:
            method_folder = 'lambdamart'
        results_root = os.path.join('../all_results', 'subqueries', 'cross_training', label_type, method_folder)
        if not os.path.exists(results_root):
            os.makedirs(results_root)
        trainging_fn = os.path.join(results_root, 'train_%s_%d' % (test_collection, query_length))
        SubqueriesLearning.write_combined_feature_fn(results_root, train, trainging_fn, query_length, True, label_type)
        testing_fn = os.path.join(results_root, 'test_%s_%d' % (test_collection, query_length))
        SubqueriesLearning.write_combined_feature_fn(results_root, test, testing_fn, query_length, False, label_type)

        if method == 1:
            for c in range(-3, 5):
                model_output_fn = os.path.join(results_root, 'model_%s_%d_%s' 
                    % (test_collection, query_length, str(10**c)) )
                if not os.path.exists(model_output_fn):
                    command = ['svm_rank_learn -c %s %s %s' % (str(10**c), trainging_fn, model_output_fn)]
                    p = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
                    returncode = p.wait()
                    out, error = p.communicate()
                    if returncode != 0:
                        print "Run Query Error: %s %s" % (command, error)
                        continue

                predict_fn = os.path.join(results_root, 'predict_%s_%d_%s' 
                    % (test_collection, query_length, str(10**c)))
                if not os.path.exists(predict_fn):
                    command = ['svm_rank_classify %s %s %s' 
                        % (testing_fn, model_output_fn, predict_fn)]
                    p = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
                    returncode = p.wait()
                    out, error = p.communicate()
                    if returncode != 0:
                        raise NameError("Run Query Error: %s %s" % (command, error))
        elif method == 2:
            for leaf in range(2, 10):
                model_output_fn = os.path.join(results_root, 'model_%s_%d_%d' 
                    % (test_collection, query_length, leaf) )
                if not os.path.exists(model_output_fn):
                    command = ['java -jar -Xmx2g ~/Downloads/RankLib-2.8.jar -train %s -ranker 6 -leaf %d -save %s' % (trainging_fn, leaf, model_output_fn)]
                    p = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
                    returncode = p.wait()
                    out, error = p.communicate()
                    if returncode != 0:
                        print "Run Query Error: %s %s" % (command, error)

                predict_fn = os.path.join(results_root, 'predict_%s_%d_%d' 
                    % (test_collection, query_length, leaf))
                if not os.path.exists(predict_fn):
                    command = ['java -jar -Xmx2g ~/Downloads/RankLib-2.8.jar -load %s -rank %s -score %s' 
                        % (model_output_fn, testing_fn, predict_fn)]
                    p = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
                    returncode = p.wait()
                    out, error = p.communicate()
                    if returncode != 0:
                        raise NameError("Run Query Error: %s %s" % (command, error))
        

    ##########
    # MI Learn
    ##########
    def load_gt_optimal(self, qids=[], method='okapi'):
        q_class = Query(self.corpus_path)
        queries = {ele['num']:ele['title'] for ele in q_class.get_queries()}      
        r = {}
        for_sort = []
        for qid in qids:
            p = []
            orig_query = queries[qid]
            with open(os.path.join(self.collected_results_root, qid)) as f:
                csvr = csv.reader(f)
                for row in csvr:
                    subquery_id = row[0]
                    subquery = row[1]
                    model_para = row[2]
                    ap = float(row[3])
                    if method in model_para:
                        p.append((subquery_id, ap))
                        if subquery_id == str(len(orig_query.split()))+'-0':
                            p_all_term = ap
            if p:
                p.sort(key=itemgetter(1), reverse=True)
                r[qid] = {'max': p[0], 'diff': p[0][1]-p_all_term}
                for_sort.append((qid, p[0][0], p[0][1]-p_all_term))
        for_sort.sort(key=itemgetter(1), reverse=True)
        return r, for_sort

    def gen_top2_subqueries(self):
        q_class = Query(self.corpus_path)
        queries = {ele['num']:ele['title'] for ele in q_class.get_queries()}
        method = 'okapi'
        r = []
        for qid in queries.keys():
            p = []
            orig_query = queries[qid]
            qlen = len(orig_query.split())
            if qlen < 2:
                continue
            with open(os.path.join(self.collected_results_root, qid)) as f:
                csvr = csv.reader(f)
                for row in csvr:
                    subquery_id = row[0]
                    subquery = row[1]
                    model_para = row[2]
                    ap = float(row[3])
                    if method in model_para:
                        if subquery_id == str(qlen)+'-0':
                            p_all_term = ap
                        else:
                            p.append((subquery_id, subquery, ap))       
            if p:
                p.sort(key=itemgetter(2), reverse=True)
                r.append((qid, qlen,orig_query, p_all_term, p[0][1], p[0][2], p[1][1], p[1][2], p[0][2]- < 1e-6))
        results_root = os.path.join('../all_results', 'subqueries', 'top2_subqueries')
        if not os.path.exists(results_root):
            os.makedirs(results_root)
        with open(os.path.join(results_root, self.collection_name+'.csv'), 'wb') as f:
            w = csv.writer(f)
            w.writerows(r)


    def mi_learn_algo(self, mi_vec, thres=1.0):
        """
        We start from query length of 3 ...
        [TODO] query length > 3
        """
        cluster = [[]]
        cluster[-1].append(mi_vec[0])
        i = 1
        while True:
            while i < len(mi_vec) and mi_vec[i][1] - thres < mi_vec[i-1][1]:
                cluster[-1].append(mi_vec[i])
                i += 1
            if i < len(mi_vec):
                cluster.append([])
                cluster[-1].append(mi_vec[i])
                i += 1
            if i >= len(mi_vec):
                break
        if len(cluster) == 3:
            return cluster[-1][0][0], 1
        elif len(cluster) == 1:
            return '3-0', 2
        else:
            if len(cluster[0]) == 1:
                return cluster[0][0][0], 3
            elif len(cluster[0]) == 2:
                return cluster[0][1][0], 4
                #return '3-0'

    def cluster_subqueries(self, query_length=3, mi_distance=5, thres=1.0):
        """
        mi: The distance of mutual information. It decides 
        which mutual information will be used to compute - 
        either 1,5,10,20,50,100.
        """
        q = Query(self.corpus_path)
        if query_length == 0:
            queries = q.get_queries()
        else:
            queries = q.get_queries_of_length(query_length)
        queries = {ele['num']:ele['title'] for ele in queries}

        all_features = {}
        gt_optimal, diff_sorted_qid = self.load_gt_optimal(queries.keys())
        for qid in os.listdir(self.subqueries_mapping_root):
            if qid not in queries:
                continue
            all_features[qid] = []
            mi_features_root = os.path.join(self.subqueries_features_root, self.feature_mapping[1])
            with open(os.path.join(mi_features_root, qid)) as f:
                qid_features = json.load(f)
            for subquery_id in sorted(qid_features, key=self.sort_subquery_id):
                if subquery_id.split('-')[0] == '2': # we only need pairwise mi
                    all_features[qid].append((subquery_id, qid_features[subquery_id][str(mi_distance)][0]))
        
        results = {}
        patterns = {}
        for qid, diff in diff_sorted_qid:
            # print '-'*30
            # print qid
            all_features[qid].sort(key=itemgetter(1))
            results[qid], _type = self.mi_learn_algo(all_features[qid], thres)
            print qid, all_features[qid], results[qid], _type, gt_optimal[qid] if qid in gt_optimal else None
            raw_input()
            if _type not in patterns:
                patterns[_type] = {'predict': {}, 'gt': {}}
            if results[qid] not in patterns[_type]['predict']:
                patterns[_type]['predict'][results[qid]] = 0
            patterns[_type]['predict'][results[qid]] += 1
            if qid in gt_optimal:
                opt_subquery_id = gt_optimal[qid]['max'][0]
                if opt_subquery_id not in patterns[_type]['gt']:
                    patterns[_type]['gt'][opt_subquery_id] = 0
                patterns[_type]['gt'][opt_subquery_id] += 1
        print json.dumps(patterns, indent=2)
        #exit()    
        self.evaluate_learn(results)

    def evaluate_learn(self, results, method='okapi'):
        ap_arr = []
        for qid in results:
            with open(os.path.join(self.collected_results_root, qid)) as f:
                csvr = csv.reader(f)
                for row in csvr:
                    subquery_id = row[0]
                    subquery = row[1]
                    model_para = row[2]
                    ap = float(row[3])
                    if method in model_para and results[qid] == subquery_id:
                        ap_arr.append(ap)
                        break
        print np.mean(ap_arr)

    @staticmethod
    def gen_resources_for_crowdsourcing_batch(collection_paths_n_names, query_length=0):
        # 1. get qids from all collections where the AP gap between the 
        # optimal subquery and original query is sorted in descending order
        all_qids = []
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
                    ele = list(ele)
                    ele.insert(0, collection_name)
                    ele.insert(0, collection_path)
                    all_qids.append(ele)
        all_qids.sort(key=itemgetter(4), reverse=True)
        results_root = os.path.join('../all_results', 'subqueries', 'crowdsourcing')
        if not os.path.exists(results_root):
            os.makedirs(results_root)
        with open(os.path.join(results_root, 'gt.csv'), 'wb') as f:
            cw = csv.writer(f)
            cw.writerows(all_qids)
        return all_qids 

    def dump_doc(self, fn, rel_docs, output_dir, subquery=True):
        with open(fn) as f:
            first_100_lines = [line.strip() for line in f.readlines()[:100]]
        doc_dir = os.path.join(output_dir, 'docs')
        if not os.path.exists(doc_dir):
            os.makedirs(doc_dir)
        if subquery:
            runfile_fn = os.path.join(output_dir, 'runfile_subquery')
        else:
            runfile_fn = os.path.join(output_dir, 'runfile_allterm')
        with open(runfile_fn, 'wb') as f:
            for line in first_100_lines:
                qid, tf_details, docid, rank, score, doc_details = line.split()
                if not os.path.exists(os.path.join(doc_dir, docid)):
                    dumpindex = '~/usr/indri-5.11/bin/dumpindex' if self.collection_name == 'GOV2' else 'dumpindex_EX'
                    index = 'index_indri511' if self.collection_name == 'GOV2' else 'index'
                    subprocess.call(['%s %s/%s dt `%s %s/%s di docno %s` > %s' 
                        % (dumpindex, self.corpus_path, index, dumpindex, self.corpus_path, index, docid, os.path.join(doc_dir, docid))], shell=True)
                f.write('%s %s %s %d %s %s\n' % (qid, tf_details, docid, 1 if docid in rel_docs else 0, score, doc_details))

    def get_queries(self):
        query_file_path = os.path.join(self.corpus_path, 'raw_topics')
        with open(query_file_path) as f:
            s = f.read()
            all_topics = re.findall(r'<top>.*?<\/top>', s, re.DOTALL)
            #print all_topics
            #print len(all_topics)

            _all = []
            for t in all_topics:
                t = re.sub(r'<\/.*?>', r'', t, flags=re.DOTALL)
                a = re.split(r'(<.*?>)', t.replace('<top>',''), re.DOTALL)
                #print a
                aa = [ele.strip() for ele in a if ele.strip()]
                d = {}
                for i in range(0, len(aa), 2):
                    """
                    if i%2 != 0:
                        if aa[i-1] == '<num>':
                            aa[i] = aa[i].split()[1]
                        d[aa[i-1][1:-1]] = aa[i].strip().replace('\n', ' ')
                    """
                    tag = aa[i][1:-1]
                    value = aa[i+1].replace('\n', ' ').strip().split(':')[-1].strip()
                    if tag == 'num':
                        value = str(int(value)) # remove the trailing '0' at the beginning
                    d[tag] = value
                _all.append(d)
        return _all

    def gen_resources_for_crowdsourcing_atom(self, qid, optimal_subquery_id, ap_diff):
        q_class = Query(self.corpus_path)
        raw_queries = {ele['num']: {
            'title':ele['title'],
            'desc':ele['desc'],
            'narr':ele['narr']
        } for ele in self.get_queries()}
        queries = {ele['num']: {
            'title':ele['title'],
            'desc':ele['desc'],
            'narr':ele['narr']
        } for ele in q_class.get_queries()}
        orig_query = queries[qid]['title']
        desc_query = raw_queries[qid]['desc']
        narr_query = raw_queries[qid]['narr']
        allterm_subquery_id = str(len(orig_query.split()))+'-0'
        with open(os.path.join(self.subqueries_mapping_root, qid)) as f:
            subquery_mapping = json.load(f)
        subquery = subquery_mapping[optimal_subquery_id]
        rel_docs = Judgment(self.corpus_path).get_relevant_docs_of_some_queries([qid], format='dict')[qid]
        cs = CollectionStats(self.corpus_path)
        terms_stats = {}
        for term in orig_query.split():
            terms_stats[term] = cs.get_term_stats(term)

        all_runfiles = os.listdir(self.subqueries_runfiles_root)
        optimal_subquery_runfile = [fn for fn in all_runfiles if fn.startswith(qid+'_'+optimal_subquery_id+'_method:okapi')][0]
        allterm_subquery_runfile = [fn for fn in all_runfiles if fn.startswith(qid+'_'+allterm_subquery_id+'_method:okapi')][0]
        with open(os.path.join(self.subqueries_performance_root, optimal_subquery_runfile)) as f:
            subquery_ap = float(f.read().split()[-1])
        with open(os.path.join(self.subqueries_performance_root, allterm_subquery_runfile)) as f:
            allterm_ap = float(f.read().split()[-1])    
        info = {
            'collection': self.collection_name,
            'optimal_subqid': optimal_subquery_id,
            'allterm_subqid': allterm_subquery_id,
            'orig_query': {'title': orig_query, 'desc': desc_query, 'narr':narr_query},
            'optimal_subquery': subquery,
            'allterm_ap': allterm_ap,
            'subquery_ap': subquery_ap,
            'terms_stats': terms_stats,
            'subquery_runfile': 'runfile_subquery',
            'allterm_runfile': 'runfile_allterm'
        }
        results_root = os.path.join('../all_results', 'subqueries', 'crowdsourcing', self.collection_name+'_'+qid)
        if not os.path.exists(results_root):
            os.makedirs(results_root)
        self.dump_doc(os.path.join(self.subqueries_runfiles_root, optimal_subquery_runfile), 
            rel_docs, results_root, True)
        self.dump_doc(os.path.join(self.subqueries_runfiles_root, allterm_subquery_runfile), 
            rel_docs, results_root, False)
        with open( os.path.join(results_root, 'info.json'), 'wb' ) as f:
            json.dump(info, f, indent=2, sort_keys=True)
        
