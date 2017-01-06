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
            10: 'QLEN'
        }

        self.svm_model_root = os.path.join(self.output_root, 'svm_rank', 'models')
        if not os.path.exists(self.svm_model_root):
            os.makedirs(self.svm_model_root)
        self.svm_predict_root = os.path.join(self.output_root, 'svm_rank', 'predict')
        if not os.path.exists(self.svm_predict_root):
            os.makedirs(self.svm_predict_root)

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
            elif feature_idx >= 9: # query length
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
                    elif feature_idx >= 9: # query length
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
                    ap = first_line.split()[-1]
            except:
                continue
            if qid not in results:
                results[qid] = {}
            if model in model_para:
                results[qid][subquery_id] = ap

        return results

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

    def batch_gen_svm_rank_paras(self):
        paras = []
        for fn in os.listdir(os.path.join(self.subqueries_features_root, 'final')):
            for c in range(-3, 5):
                if not os.path.exists(os.path.join(self.svm_model_root, fn+'_'+str(10**c))):
                    paras.append((self.corpus_path, self.collection_name, fn, c))
        return paras

    def svm_rank_wrapper(self, query_length, c):
        command = ['svm_rank_learn', '-c', str(10**c), 
            os.path.join(self.subqueries_features_root, 'final', query_length), 
            os.path.join(self.svm_model_root, query_length+'_'+str(10**c))]
        subprocess.call(command)

    def evaluate_svm_model(self):
        all_models = {}
        error_rate_fn = os.path.join(self.output_root, 'svm_rank', 'err_rate')
        error_rates = {}
        for fn in os.listdir(self.svm_model_root):
            predict_output_fn = os.path.join(self.svm_predict_root, fn)
            if os.path.exists(predict_output_fn) and os.path.exists(error_rate_fn):
                continue
            query_length = fn.split('_')[0]
            c = fn.split('_')[1]
            command = ['svm_rank_classify %s %s %s' 
                % (os.path.join(self.subqueries_features_root, 'final', query_length), 
                    os.path.join(self.svm_model_root, fn), 
                    os.path.join(self.svm_predict_root, fn))]
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
        for query_length in sorted(all_models):
            # first sort based on err_rate
            all_models[query_length].sort(key=itemgetter(1))

            # model prediction performance related
            svm_predict_optimal_subquery_len_dist[query_length] = {}
            predict_optimal_performance = {}
            existing_performance = {}
            correct_cnt = 0.0
            incorrect_cnt = 0.0
            optimal_ground_truth = 0.0
            optimal_svm_predict = 0.0
            performance_using_all_terms = 0.0
            fn = all_models[query_length][0][0]
            feature_fn = os.path.join(self.subqueries_features_root, 'final', str(query_length))
            predict_fn = os.path.join(self.svm_predict_root, fn)
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
            with open(os.path.join(self.final_output_root, self.collection_name+'-svm_subquery_dist.md'), 'wb') as f:
                f.write('### %s\n' % (self.collection_name))

                f.write('#### all optimals\n')
                f.write('| using all terms | optimal (ground truth) | svm optimal |\n')
                f.write('|--------|--------|--------|\n')
                f.write('| %.4f | %.4f | %.4f |\n' 
                    % (performance_using_all_terms/query_cnt, 
                        optimal_ground_truth/query_cnt, 
                        optimal_svm_predict/query_cnt))

                f.write('\n#### svm predict subquery length distribution\n')
                f.write('| | | | | |\n')
                f.write('|--------|--------|--------|--------|--------|\n')
                for query_len in svm_predict_optimal_subquery_len_dist:
                    f.write('| %d |' % (query_len))
                    for subquery_len in svm_predict_optimal_subquery_len_dist[query_len]:
                        f.write(' %d:%d |' % (subquery_len, svm_predict_optimal_subquery_len_dist[query_len][subquery_len]))
                    f.write('\n')
        
            # feature ranking related
            model_fn = all_models[query_length][0][0]
            with open(os.path.join(self.svm_model_root, model_fn)) as f:
                model = f.readlines()[-1]
            feature_weights = [(int(ele.split(':')[0]), float(ele.split(':')[1])) for ele in model.split()[1:-1]]
            feature_weights.sort(key=itemgetter(1, 0), reverse=True)
            output_root = os.path.join(self.output_root, 'svm_rank', 'featurerank')
            if not os.path.exists(output_root):
                os.makedirs(output_root)
            with open(os.path.join(output_root, str(query_length)), 'wb') as f:
                for ele in feature_weights:
                    f.write('%s: %f\n' % (feature_mapping[ele[0]], ele[1]))

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
                        optimal_svm_predict += optimal_svm_predict
                        subquery_len = int(predict_optimal_performance[qid][0][0].split('-')[0])
                        # if subquery_len not in svm_predict_optimal_subquery_len_dist[query_length]:
                        #     svm_predict_optimal_subquery_len_dist[query_length][subquery_len] = 0
                        # svm_predict_optimal_subquery_len_dist[query_length][subquery_len] += 1 
                    collection_predict_performance[collection_name] = collection_predict       
                all_performances[query_length].append((c, optimal_svm_predict, collection_predict_performance))
            all_performances[query_length].sort(key=itemgetter(1), reverse=True)
        for query_length in all_performances:
            print json.dumps(all_performances[query_length][0], indent=2)



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

