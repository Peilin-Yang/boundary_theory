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
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from query import Query
from evaluation import Evaluation
from judgment import Judgment
from performance import Performances
from collection_stats import CollectionStats
from run_subqueries import RunSubqueries
from subqueries_learning import SubqueriesLearning

class SubqueriesClassification(SubqueriesLearning):
    """
    learning the subqueries: features generation, learning, etc.
    """
    def __init__(self, path, corpus_name):
        super(SubqueriesClassification, self).__init__(path, corpus_name)

        self.term_stats_mapping = {
            'CTF': 'total_occur',
            'DF': 'df',
            'LOGIDF': 'log(idf1)',
            'MAXTF': 'maxTF',
            'MINTF': 'minTF',
            'AVGTF': 'avgTF',
            'VARTF': 'varTF'
        }

    def get_term_stats(self, qid, feature_formal_name, required_feature):
        features_root = os.path.join(self.subqueries_features_root, feature_formal_name)
        cs = CollectionStats(self.corpus_path)
        q = Query(self.corpus_path)
        queries = q.get_queries()
        queries = {ele['num']:ele['title'] for ele in queries}
        features = {}
        for term in queries[qid].split():
            features[term] = cs.get_term_stats(term)[required_feature]

        return features

    def get_all_sorts_features(self, feature_vec, 
            allowed=['min', 'max', 'max-min', 'max/min', 'mean', 'std', 'sum', 'gmean']):
        _all = {
            'min': np.min(feature_vec) if feature_vec else 0.0, 
            'max': np.max(feature_vec) if feature_vec else 0.0,  
            'max-min': np.max(feature_vec)-np.min(feature_vec) if feature_vec else 0.0, 
            'max/min': np.max(feature_vec)/np.min(feature_vec) if feature_vec and np.min(feature_vec) != 0 else 0,
            'mean': np.mean(feature_vec) if feature_vec else 0.0,
            'std': np.std(feature_vec) if feature_vec else 0.0, 
            'sum': np.sum(feature_vec) if feature_vec else 0.0, 
            'gmean': 0 if np.isnan(scipy.stats.mstats.gmean(feature_vec)) else scipy.stats.mstats.gmean(feature_vec)
        }
        return {k:v for k,v in _all.items() if k in allowed}

    def load_term_scores(self, runfile, model_para, method=1, cutoff=50):
        cs = CollectionStats(self.corpus_path)
        all_scores = []
        with open(runfile) as f:
            line_idx = 0
            for line in f:
                line = line.strip()
                if line:
                    row = line.split()
                    tf_details = row[1]
                    terms = [ele.split('-')[0] for ele in tf_details.split(',')]
                    tfs = [float(ele.split('-')[1]) for ele in tf_details.split(',')]
                    dl = float(row[-1].split(',')[0].split(':')[1])
                    if method == 1: # simple TF
                        scores = tfs
                    elif method == 2: # BM25
                        scores = [tf*cs.get_term_logidf1(terms[i])*2.2/(tf+1.2*(1-model_para+model_para*dl/cs.get_avdl())) for i, tf in enumerate(tfs)]
                    all_scores.append(scores)
                line_idx += 1
                if line_idx >= cutoff:
                    break
        return all_scores

    def ranking_scores_features(self, qid):
        methods = ['okapi']
        optimal_performances = Performances(self.corpus_path).load_optimal_performance(methods)[0]
        indri_model_para = 'method:%s,' % optimal_performances[0] + optimal_performances[2]
        model_para = float(optimal_performances[2].split(':')[1])
        with open(os.path.join(self.subqueries_mapping_root, qid)) as f:
            subquery_mapping = json.load(f)
        features = {}
        features_fn = os.path.join(self.subqueries_features_root, 'CLT', qid)
        with open(features_fn) as f:
            j = json.load(f)
            for subquery_id, subquery_str in subquery_mapping.items():
                features[subquery_id] = j[subquery_id]['50']
        return features

    def get_classification_feature_mapping(self, query_len=0):
        if query_len == 3:
            pass      

    def batch_gen_query_classification_features_paras(self):
        q = Query(self.corpus_path)
        queries = q.get_queries()
        queries = {ele['num']:ele['title'] for ele in queries}
        output_root = os.path.join(self.subqueries_features_root, 'classification', 'qids')
        if not os.path.exists(output_root):
            os.makedirs(output_root)
        all_paras = []
        for qid in queries:
            output_fn = os.path.join(output_root, qid)
            if not os.path.exists(output_fn):
                all_paras.append((self.corpus_path, self.collection_name, qid, queries[qid]))
        return all_paras

    def gen_query_classification_features(self, qid, query_str):
        features = {}

        # term features
        terms_stats = {}
        for k,v in self.term_stats_mapping.items():
            terms_stats[k] = self.get_term_stats(qid, k, v)
        relations = ['max', 'min', 'mean', 'std']
        for k in self.term_stats_mapping:
            features[k] = self.get_all_sorts_features(terms_stats[k].values(), relations)

        # term scores features
        terms_scores = self.ranking_scores_features(qid)
        terms_scores_fl = {}
        for subquery_id in terms_scores:
            subquery_len = subquery_id.split('-')[0]
            if not subquery_len in terms_scores_fl:
                terms_scores_fl[subquery_len] = []
            terms_scores_fl[subquery_len].append(terms_scores[subquery_id][1]) # std
        for subquery_len in terms_scores_fl:
            if int(subquery_len) == len(query_str.split()):
                features['scores_'+subquery_len] = {'all': terms_scores_fl[subquery_len][0]}
            else:
                features['scores_'+subquery_len] = self.get_all_sorts_features(terms_scores_fl[subquery_len], relations) 
        
        output_root = os.path.join(self.subqueries_features_root, 'classification', 'qids')
        output_fn = os.path.join(output_root, qid)
        with open(output_fn, 'wb') as f:
            json.dump(features, f, indent=2)


    def output_features_classification(self, query_len=0):
        """
        Output the features only for the original queries.
        We only would like to know whether we should use original 
        query or the sub-query
        """
        output_root = os.path.join(self.subqueries_features_root, 'classification')
        if not os.path.exists(output_root):
            os.makedirs(output_root)
        output_fn = os.path.join(output_root, str(query_len))
        all_performances = self.get_all_performances()
        all_features_matrix = []
        classification = {}

        q = Query(self.corpus_path)
        if query_len == 0:
            queries = q.get_queries()
        else:
            queries = q.get_queries_of_length(query_len)
        queries = {int(ele['num']):ele['title'] for ele in queries}
        for qid in sorted(queries):
            qid = str(qid)
            if qid not in all_performances:
                continue
            qid_feature_fn = os.path.join(self.subqueries_features_root, 'classification', 'qids', qid)
            with open(qid_feature_fn) as f:
                qid_feature = json.load(f)
            all_features_matrix.append([])
            for ele in qid_feature.values():
                all_features_matrix[-1].extend(ele.values())
            sorted_subqueryid = sorted(all_performances[qid].items(), key=itemgetter(1), reverse=True)
            if int(sorted_subqueryid[0][0].split('-')[0]) != query_len:
                classification[qid] = 0
            else:
                classification[qid] = 1
            normalized = normalize(all_features_matrix, axis=0) # normalize each feature
        idx = 0
        with open(output_fn, 'wb') as f: 
            for qid in sorted(classification, key=self.sort_qid):
                f.write('%d qid:%s %s # %s\n' % (classification[qid], qid, 
                    ' '.join(['%d:%f' % (i, normalized[idx][i-1]) for i in range(1, len(normalized[idx])+1)]), 
                    str(query_len)+'-0'))
                idx += 1

    def batch_run_classification_paras(self):
        methods = {
            'svm': [10**i for i in range(-5, 5)],
            'nn': [10**i for i in range(-5, 0, 1)],
            'dt': range(1, 6)
        }
        run_paras = []
        classification_results_root = os.path.join(self.output_root, 'classification', 'results')
        if not os.path.exists(classification_results_root):
            os.makedirs(classification_results_root)
        feature_root = os.path.join(self.subqueries_features_root, 'classification')
        for query_len in os.listdir(feature_root):
            if not os.path.isfile(os.path.join(feature_root, query_len)):
                continue
            for method, paras in methods.items():
                for para in paras:
                    output_fn = os.path.join(classification_results_root, query_len+'_'+method+'_'+str(para))
                    if not os.path.exists(output_fn):
                        run_paras.append((self.corpus_path, self.collection_name, query_len, method, para))
        return run_paras

    @staticmethod
    def read_classification_features(fn):
        features = []
        classes = []
        qids = []
        with open(fn) as f:
            for line in f:
                line = line.strip()
                if line:
                    row = line.split()
                    features.append([float(ele.split(':')[1]) for ele in row[2:-2]])
                    classes.append(int(row[0]))
                    qids.append(row[1].split(':')[1])
        return features, classes, qids


    @staticmethod
    def write_combined_feature_classification_fn(results_root, l, ofn, query_length=2, reorder_qid=False):
        trainging_fn = os.path.join(results_root, 'train_%d' % query_length)
        if os.path.exists(ofn):
            os.remove(ofn)
        with open(ofn, 'ab') as f:
            qid_idx = 1
            qid_lines = {}
            for ele in l:
                collection_path = ele[0]
                collection_name = ele[1]
                feature_fn = os.path.join(collection_path, 'subqueries', 'features', 'classification', str(query_length))
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
    def cross_run_classification(train, test, query_length=2):
        results_root = os.path.join('../all_results', 'subqueries', 'cross_classification')
        if not os.path.exists(results_root):
            os.makedirs(results_root)
        test_collection = test[0][1]
        trainging_fn = os.path.join(results_root, 'train_%s_%d' % (test_collection, query_length))
        SubqueriesClassification.write_combined_feature_classification_fn(results_root, train, trainging_fn, query_length, True)
        testing_fn = os.path.join(results_root, 'test_%s_%d' % (test_collection, query_length))
        SubqueriesClassification.write_combined_feature_classification_fn(results_root, test, testing_fn, query_length, False)
        
        methods = {
            'svm': [10**i for i in range(-5, 5)],
            'nn': [10**i for i in range(-5, 0, 1)],
            'dt': range(1, 6)
        }
        for method, paras in methods.items():
            for para in paras:
                output_fn = os.path.join(results_root, 'predict_'+test_collection+'_'+str(query_length)+'_'+method+'_'+str(para))
                #if not os.path.exists(output_fn):
                train_features, train_classes, train_qids = \
                    SubqueriesClassification.read_classification_features(trainging_fn)
                testing_features, testing_classes, testing_qids = \
                    SubqueriesClassification.read_classification_features(testing_fn)
                if method == 'nn':
                    clf = MLPClassifier(solver='lbfgs', alpha=para, random_state=1)
                elif method == 'svm':
                    clf = SVC(C=para)
                elif method == 'dt':
                    clf = tree.DecisionTreeClassifier(max_depth=para)
                clf.fit(train_features, train_classes)
                predicted = clf.predict(testing_features)
                acc = accuracy_score(testing_classes, predicted)
                optimal_ground_truth, using_all_terms, second_optimal = \
                    SubqueriesClassification.load_optimal_ground_truth(test[0][0], testing_qids)
                predicted_map = {}
                for i, qid in enumerate(testing_qids):
                    if predicted[i] == 1:
                        predicted_map[qid] = using_all_terms[qid]
                    elif predicted[i] == 0 and testing_classes[i] == 0:
                        predicted_map[qid] = optimal_ground_truth[qid]
                    elif predicted[i] == 0 and testing_classes[i] == 1:
                        predicted_map[qid] = second_optimal[qid]
                with open(output_fn, 'wb') as f:
                    f.write('acc:%.2f\n' % acc)
                    f.write('predict_ap:%.4f\n' % np.mean(predicted_map.values()))

    @staticmethod
    def evaluate_cross_classification(all_data, query_length=2):
        data_mapping = {d[1]:d[0] for d in all_data}
        results_root = os.path.join('../all_results', 'subqueries', 'cross_classification')
        all_predict_data = {}
        for fn in os.listdir(results_root):
            m = re.search(r'^predict_(.*?)_(.*?)_(.*)_(.*)$', fn)
            if m:
                collection_name = m.group(1)
                query_length = int(m.group(2))
                method = m.group(3)+'_'+m.group(4)
                if query_length not in all_predict_data:
                    all_predict_data[query_length] = {}
                if method not in all_predict_data[query_length]:
                    all_predict_data[query_length][method] = []
                with open(os.path.join(results_root, fn)) as f:
                    lines = f.readlines()
                    accuracy = float(lines[0].strip().split(':')[1])
                    ap = float(lines[1].strip().split(':')[1])
                all_predict_data[query_length][method].append((collection_name, accuracy, ap))

        avg_predict_data = {}
        for query_length in all_predict_data:
            avg_predict_data[query_length] = []
            for method in all_predict_data[query_length]:
                avg_predict_data[query_length].append((method, np.mean([ele[1] for ele in all_predict_data[query_length][method]])))

        for query_length in avg_predict_data:
            avg_predict_data[query_length].sort(key=itemgetter(1), reverse=True)
        for query_length in avg_predict_data:
            method = avg_predict_data[query_length][0]
            print query_length, method, all_predict_data[query_length][method[0]]
