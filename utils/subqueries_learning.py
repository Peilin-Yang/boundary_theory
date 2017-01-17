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
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

from query import Query
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
            12: 'AVGTFCTF'
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

    def get_all_sorts_features(self, feature_vec):
        return [np.min(feature_vec), np.max(feature_vec), 
                np.max(feature_vec)-np.min(feature_vec),
                np.max(feature_vec)/np.min(feature_vec) if np.min(feature_vec) != 0 else 0,
                np.mean(feature_vec), np.std(feature_vec), 
                np.sum(feature_vec), 
                0 if np.isnan(scipy.stats.mstats.gmean(feature_vec)) else scipy.stats.mstats.gmean(feature_vec)]


    def sort_qid(self, qid):
        return int(qid)
    def sort_subquery_id(self, subquery_id):
        return int(subquery_id.split('-')[0])+float(subquery_id.split('-')[1])/10.0


    def get_feature_mapping(self):
        mapping = {}
        idx = 1
        features = ['min', 'max', 'max-min', 'max/min', 'mean', 'std', 'sum', 'gmean']
        for feature_idx, feature_name in self.feature_mapping.items():
            if feature_idx == 1: # mutual information
                withins = [1, 5, 10, 20, 50, 100]
                for w in withins:
                    for fa in features:
                        mapping[idx] = feature_name+str(w)+'('+fa+')'
                        idx += 1
            elif feature_idx == 9 or feature_idx == 10: # query length and Clarity
                mapping[idx] = feature_name
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
                        withins = [1, 5, 10, 20, 50, 100]
                        for w in withins:
                            str_w = str(w)
                            all_features[qid][subquery_id].extend(qid_features[subquery_id][str_w])
                    elif feature_idx == 9 or feature_idx == 10: # query length and Clarity
                        all_features[qid][subquery_id].append(qid_features[subquery_id])
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

    def output_features_kendallstau(self, query_len=0):
        """
        output the kendallstau between features and the ranking of subqueries.
        The output can be used to reduce the dimension of the feature space
        """
        output_root = os.path.join(self.subqueries_features_root, 'kendallstau')
        if not os.path.exists(output_root):
            os.makedirs(output_root)
        output_fn = os.path.join(output_root, str(query_len))
        feature_mapping = self.get_feature_mapping()
        all_performances = self.get_all_performances()
        all_features = self.get_all_features(query_len)
        all_features_matrix = []
        kendallstau = {}
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
    def output_features_kendallstau_all_collection(collection_paths_n_names, query_length=0):
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
            with open(os.path.join(collection_path, 'subqueries', 'features', 'kendallstau', str(query_length))) as f:
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
        feature_mapping = self.get_feature_mapping()
        all_performances = self.get_all_performances()
        all_features = self.get_all_features(query_len)
        all_features_matrix = []
        classification_features = {}
        classification = {}
        for qid in sorted(all_performances): 
            if qid not in all_features:
                continue
            classification_features[qid] = all_features[qid][str(query_len)+'-0']
            all_features_matrix.append(all_features[qid][str(query_len)+'-0'])
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
            'nn': [10**i for i in range(-5, 0, 1)]
        }
        print methods
        run_paras = []
        classification_results_root = os.path.join(self.output_root, 'classification', 'results')
        if not os.path.exists(classification_results_root):
            os.makedirs(classification_results_root)
        feature_root = os.path.join(self.subqueries_features_root, 'classification')
        for query_len in os.listdir(feature_root):
            for method, paras in methods.items():
                for para in paras:
                    output_fn = os.path.join(classification_results_root, query_len+'_'+method+'_'+str(para))
                    if not os.path.exists(output_fn):
                        run_paras.append((self.corpus_path, self.collection_name, query_len, method, para))
        return run_paras

    def read_classification_features(fn):
        classes = []
        with open(fn) as f:
            for line in f:
                line = line.strip()
                if line:
                    row = line.split()
                    features.append([float(ele.split(':')[1]) for ele in row[2:-2]])
                    classes.append(int(row[0]))
        return features, classes


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
        SubqueriesLearning.write_combined_feature_classification_fn(results_root, train, trainging_fn, query_length, True)
        testing_fn = os.path.join(results_root, 'test_%s_%d' % (test_collection, query_length))
        SubqueriesLearning.write_combined_feature_classification_fn(results_root, test, testing_fn, query_length, False)
        
        methods = {
            'svm': [10**i for i in range(-5, 5)],
            'nn': [10**i for i in range(-5, 0, 1)]
        }
        for method, paras in methods.items():
            for para in paras:
                output_fn = os.path.join(results_root, 'predict_'+str(query_length)+'_'+method+'_'+str(para))
                if not os.path.exists(output_fn):
                    train_features, train_classes = 
                        SubqueriesLearning.read_classification_features(trainging_fn)
                    testing_features, testing_classes = 
                        SubqueriesLearning.read_classification_features(testing_fn)
                    if method == 'nn':
                        clf = MLPClassifier(solver='lbfgs', alpha=para, random_state=1)
                    elif method == 'svm':
                        clf = SVC(C=para)
                    clf.fit(train_features, train_classes)
                    predicted = clf.predict(testing_classes)
                    print query_length, method, para
                    print classification_report(testing_classes, predicted)
                    exit()

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
        with open(os.path.join(output_root, str(query_len)), 'wb') as f: 
            for qid in sorted(all_features, key=self.sort_qid):
                for subquery_id in sorted(all_features[qid], key=self.sort_subquery_id):
                    ### sample training: "3 qid:1 1:1 2:1 3:0 4:0.2 5:0 # 1A"
                    if qid in all_performances and subquery_id in all_performances[qid]:
                        f.write('%s qid:%s %s # %s\n' % (str(all_performances[qid][subquery_id]), qid, 
                            ' '.join(['%d:%f' % (i, normalized[idx][i-1]) for i in range(1, len(normalized[idx])+1)]), 
                            subquery_id))
                    idx += 1

    def batch_gen_svm_rank_paras(self, feature_type=1):
        if feature_type == 2:
            folder = 'kendallstau'
        else:
            folder = 'final'
        paras = []
        svm_model_root = os.path.join(self.output_root, 'svm_rank', folder, 'models')
        if not os.path.exists(svm_model_root):
            os.makedirs(svm_model_root)
        svm_predict_root = os.path.join(self.output_root, 'svm_rank', folder, 'predict')
        if not os.path.exists(svm_predict_root):
            os.makedirs(svm_predict_root)
        for fn in os.listdir(os.path.join(self.subqueries_features_root, folder)):
            for c in range(-5, 5):
                if not os.path.exists(os.path.join(svm_model_root, fn+'_'+str(10**c))):
                    paras.append((self.corpus_path, self.collection_name, folder, fn, c))
        return paras

    def svm_rank_wrapper(self, folder, query_length, c):
        svm_model_root = os.path.join(self.output_root, 'svm_rank', folder, 'models')
        command = ['svm_rank_learn', '-c', str(10**c), 
            os.path.join(self.subqueries_features_root, folder, query_length), 
            os.path.join(svm_model_root, query_length+'_'+str(10**c))]
        subprocess.call(command)

    def evaluate_svm_model(self, feature_type=1):
        if feature_type == 2:
            folder = 'kendallstau'
        else:
            folder = 'final'
        svm_model_root = os.path.join(self.output_root, 'svm_rank', folder, 'models')
        svm_predict_root = os.path.join(self.output_root, 'svm_rank', folder, 'predict')
        all_models = {}
        error_rate_fn = os.path.join(self.output_root, 'svm_rank', folder, 'err_rate')
        error_rates = {}
        for fn in os.listdir(svm_model_root):
            predict_output_fn = os.path.join(svm_predict_root, fn)
            if os.path.exists(predict_output_fn) and os.path.exists(error_rate_fn):
                continue
            query_length = fn.split('_')[0]
            c = fn.split('_')[1]
            command = ['svm_rank_classify %s %s %s' 
                % (os.path.join(self.subqueries_features_root, folder, query_length), 
                    os.path.join(svm_model_root, fn), 
                    os.path.join(svm_predict_root, fn))]
            p = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
            returncode = p.wait()
            out, error = p.communicate()
            if returncode != 0:
                raise NameError("Run Query Error: %s" % (command) )
            query_length = int(query_length)
            err_rate = float(out.split('\n')[-2].split(':')[1])
            error_rates[fn] = err_rate
            if query_length not in all_models:
                all_models[query_length] = []
            all_models[query_length].append((fn, err_rate))
        if error_rates:
            with open(error_rate_fn, 'wb') as f:
                json.dump(error_rates, f, indent=2)
        if not all_models:
            with open(error_rate_fn) as f:
                error_rates = json.load(f)
            for fn in error_rates:
                query_length = int(fn.split('_')[0])
                if query_length not in all_models:
                    all_models[query_length] = []
                all_models[query_length].append((fn, error_rates[fn]))

        feature_mapping = self.get_feature_mapping()
        svm_predict_optimal_subquery_len_dist = {}
        with open(os.path.join(self.final_output_root, self.collection_name+'-svm_subquery_dist-%s.md' % folder), 'wb') as ssdf:
            ssdf.write('### %s\n' % (self.collection_name))
            ssdf.write('| query len | using all terms | optimal (ground truth) | svm optimal |\n')
            ssdf.write('|--------|--------|--------|--------|\n')
            for query_length in sorted(all_models):
                # first sort based on err_rate
                all_models[query_length].sort(key=itemgetter(1))

                # model prediction performance related
                svm_predict_optimal_subquery_len_dist[query_length] = {}
                predict_optimal_performance = {}
                existing_performance = {}
                optimal_ground_truth = 0.0
                optimal_svm_predict = 0.0
                performance_using_all_terms = 0.0
                fn = all_models[query_length][0][0]
                feature_fn = os.path.join(self.subqueries_features_root, folder, str(query_length))
                predict_fn = os.path.join(svm_predict_root, fn)
                with open(predict_fn) as f:
                    predict_res = [float(line.strip()) for line in f.readlines()]
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
                    optimal_svm_predict += predict_optimal_performance[qid][0][2]
                    subquery_len = int(predict_optimal_performance[qid][0][0].split('-')[0])
                    if subquery_len not in svm_predict_optimal_subquery_len_dist[query_length]:
                        svm_predict_optimal_subquery_len_dist[query_length][subquery_len] = 0
                    svm_predict_optimal_subquery_len_dist[query_length][subquery_len] += 1

                query_cnt = len(predict_optimal_performance)
                ssdf.write('| %d | %.4f | %.4f | %.4f |\n' 
                    % ( query_length, 
                        performance_using_all_terms/query_cnt, 
                        optimal_ground_truth/query_cnt, 
                        optimal_svm_predict/query_cnt))

                # feature ranking related
                model_fn = all_models[query_length][0][0]
                with open(os.path.join(svm_model_root, model_fn)) as f:
                    model = f.readlines()[-1]
                feature_weights = [(int(ele.split(':')[0]), float(ele.split(':')[1])) for ele in model.split()[1:-1]]
                feature_weights.sort(key=itemgetter(1, 0), reverse=True)
                output_root = os.path.join(self.output_root, 'svm_rank', folder, 'featurerank')
                if not os.path.exists(output_root):
                    os.makedirs(output_root)
                with open(os.path.join(output_root, str(query_length)), 'wb') as f:
                    for ele in feature_weights:
                        f.write('%s: %f\n' % (feature_mapping[ele[0]], ele[1]))

            ssdf.write('\n#### svm predict subquery length distribution\n')
            ssdf.write('| | | | | |\n')
            ssdf.write('|--------|--------|--------|--------|--------|\n')
            for query_len in svm_predict_optimal_subquery_len_dist:
                ssdf.write('| %d |' % (query_len))
                for subquery_len in svm_predict_optimal_subquery_len_dist[query_len]:
                    ssdf.write(' %d:%d |' % (subquery_len, svm_predict_optimal_subquery_len_dist[query_len][subquery_len]))
                ssdf.write('\n')

    @staticmethod
    def write_combined_feature_fn(results_root, l, ofn, query_length=2, reorder_qid=False):
        trainging_fn = os.path.join(results_root, 'train_%d' % query_length)
        if os.path.exists(ofn):
            os.remove(ofn)
        with open(ofn, 'ab') as f:
            qid_idx = 1
            qid_lines = {}
            for ele in l:
                collection_path = ele[0]
                collection_name = ele[1]
                feature_fn = os.path.join(collection_path, 'subqueries', 'features', 'final', str(query_length))
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
    def evaluate_svm_cross_testing(all_data, query_length=2):
        data_mapping = {d[1]:d[0] for d in all_data}
        results_root = os.path.join('../all_results', 'subqueries', 'cross_training')
        all_predict_data = {}
        for fn in os.listdir(results_root):
            m = re.search(r'^predict_(.*?)_(.*?)_(.*)$', fn)
            if m:
                collection_name = m.group(1)
                query_length = int(m.group(2))
                c = m.group(3)
                if query_length not in all_predict_data:
                    all_predict_data[query_length] = {}
                if c not in all_predict_data[query_length]:
                    all_predict_data[query_length][c] = []
                all_predict_data[query_length][c].append(collection_name)

        all_performances = {}
        for query_length in all_predict_data:
            all_performances[query_length] = []
            for c in all_predict_data[query_length]:
                if len(all_predict_data[query_length][c]) != len(all_data):
                    print 'query length: %d and c: %s does not have enough data ... %d/%d' \
                        % (query_length, c, len(all_predict_data[query_length][c]), len(all_data))
                    continue
                #svm_predict_optimal_subquery_len_dist[query_length] = {}
                existing_performance = {}
                collection_predict_performance = {}
                optimal_ground_truth = 0.0
                optimal_svm_predict = 0.0
                performance_using_all_terms = 0.0
                for collection_name in all_predict_data[query_length][c]: 
                    predict_optimal_performance = {}
                    feature_fn = os.path.join(results_root, 'test_%s_%d' % (collection_name, query_length))
                    predict_fn = os.path.join(results_root, 'predict_%s_%d_%s' % (collection_name, query_length, c))
                    with open(predict_fn) as f:
                        predict_res = [float(line.strip()) for line in f.readlines()]
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
                all_performances[query_length].append((c, optimal_svm_predict, collection_predict_performance))
            all_performances[query_length].sort(key=itemgetter(1), reverse=True)
        for query_length in all_performances:
            print query_length, json.dumps(all_performances[query_length][0], indent=2)



    @staticmethod
    def cross_testing(train, test, query_length=2):
        """
        train and test are list of (collection_path, collection_name)
        """
        test_collection = test[0][1]
        results_root = os.path.join('../all_results', 'subqueries', 'cross_training')
        if not os.path.exists(results_root):
            os.makedirs(results_root)
        trainging_fn = os.path.join(results_root, 'train_%s_%d' % (test_collection, query_length))
        SubqueriesLearning.write_combined_feature_fn(results_root, train, trainging_fn, query_length, True)
        testing_fn = os.path.join(results_root, 'test_%s_%d' % (test_collection, query_length))
        SubqueriesLearning.write_combined_feature_fn(results_root, test, testing_fn, query_length, False)
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


    @staticmethod
    def write_combined_feature_fn(results_root, l, ofn, query_length=2, reorder_qid=False):
        trainging_fn = os.path.join(results_root, 'train_%d' % query_length)
        if os.path.exists(ofn):
            os.remove(ofn)
        with open(ofn, 'ab') as f:
            qid_idx = 1
            qid_lines = {}
            for ele in l:
                collection_path = ele[0]
                collection_name = ele[1]
                feature_fn = os.path.join(collection_path, 'subqueries', 'features', 'final', str(query_length))
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