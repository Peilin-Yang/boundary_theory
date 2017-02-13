import os,sys
import ast
import json
import csv
from operator import itemgetter
import shutil
import argparse
import subprocess, shlex
from subprocess import Popen, PIPE
import inspect
from inspect import currentframe, getframeinfo

import numpy as np

from utils.ArrayJob import ArrayJob
import g
from tie_breaker import TieBreaker
from utils.evaluation import Evaluation
from utils.rel_tf_stats import RelTFStats
from run_subqueries import RunSubqueries
from subqueries_learning import SubqueriesLearning

_root = '../../reproduce/collections/'
output_root = '../all_results/'
max_nodes = 120


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
    """
    for i, ele in enumerate(all_paras):
        batch_script_fn = os.path.join( batch_script_root, '%s-%d.qs' % (para_label, i) )
        batch_para_fn = os.path.join(para_dir, 'para_file_%d' % i)
        with open(batch_para_fn, 'wb') as bf:
            bf.write('\n'.join(ele))
        command = 'python %s -%s' % (
            inspect.getfile(inspect.currentframe()), \
            batch_pythonscript_para
        )
        arrayjob_script = ArrayJob.ArrayJob()
        arrayjob_script.output_batch_qs_file(batch_script_fn, command, quote_command, True, batch_para_fn, len(ele))
        run_batch_gen_query_command = 'qsub %s' % batch_script_fn
        subprocess.call( shlex.split(run_batch_gen_query_command) )
    """

def get_already_generated_qids(collection_path):
    r = []
    fn = os.path.join(collection_path, 'detailed_doc_stats_log')
    if os.path.exists( fn ):
        with open(fn) as f:
            r = [line.strip() for line in f.readlines() if line.strip()]

    return r


def collect_existing_detailed_doc_stats_results(intermediate_results_root, 
        all_results_dict, already_generated_qids, final_output_path):
    """
    collect the parallel results generated by multiple nodes
    """
    if not os.path.exists(final_output_path):
        os.makedirs(final_output_path)

    final_results_only_docid = {}
    # final_results = {}
    # final_results_only_docid = {}
    # for fn in os.listdir(final_output_path):
    #     if fn not in already_generated_qids:
    #         with open(os.path.join(final_output_path, fn)) as f:
    #             for line in f:
    #                 line = line.strip()
    #                 if line:
    #                     row = line.split(',')
    #                     docid = row[1]
    #                     row[2] = ast.literal_eval(row[2])
    #                     if fn not in final_results_only_docid:
    #                         final_results_only_docid[fn] = []
    #                     final_results_only_docid[fn].append(docid)
    #                     if fn not in final_results:
    #                         final_results[fn] = []
    #                     final_results[fn].append(row)
    #     else:
    #         print 'Skipping[final results folder] qid:%s' % fn

    #print len(final_results)

    if os.path.exists(intermediate_results_root):
        for fn in os.listdir(intermediate_results_root):
            with open(os.path.join(intermediate_results_root, fn)) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            row = line.split(',')
                            qid = row[0]
                            did = row[1]
                            if qid in already_generated_qids:
                                print 'Skipping[intermediate results folder] qid:%s' % qid
                                continue
                                
                            # row[2] = ast.literal_eval(row[2])
                            # if qid in final_results and did in final_results[qid]:
                            #     #exists
                            #     pass
                            # else:
                            #     if qid not in final_results:
                            #         final_results[qid] = []
                            #     final_results[qid].append(row)
                            #     if qid not in final_results_only_docid:
                            #         final_results_only_docid[qid] = []
                            #     final_results_only_docid[qid].append(did)
                            if qid not in final_results_only_docid:
                                final_results_only_docid[qid] = []
                            final_results_only_docid[qid].append(did)
                            with open(os.path.join(final_output_path, qid), 'ab') as f:
                                f.write(line+'\n')
                        except:
                            print line

    # for qid in final_results:
    #     final_results[qid].sort(key=itemgetter(2), reverse=True)

    # for qid in final_results:
    #     with open(os.path.join(final_output_path, qid), 'wb') as f:
    #         for ele in final_results[qid]:
    #             ele[2] = str(ele[2])
    #             f.write(','.join(ele)+'\n')
    try:
        #shutil.rmtree(intermediate_results_root)
        pass
    except:
        print 'rmtree %s failed!' % intermediate_results_root


    all_results_dict_local = {qid:[ele[0] for ele in r] for qid,r in all_results_dict.items() if qid not in already_generated_qids}
    not_run = {}
    for qid, doc_ids in all_results_dict_local.items():
        if qid not in final_results_only_docid:
            not_run[qid] = all_results_dict[qid]
        else:
            not_run_list = list(set(doc_ids)-set(final_results_only_docid[qid]))
            if not_run_list:
                if qid not in not_run:
                    not_run[qid] = []
                for doc_id in not_run_list:
                    for ele in all_results_dict[qid]:
                        if ele[0] == doc_id:
                            not_run[qid].append(ele)
                            break
        if qid not in not_run or not not_run[qid]:
            already_generated_qids.append(qid)
    return not_run


def gen_detailed_doc_stats_paras(method_name):
    all_paras = []
    for q in g.query:
        collection_name = q['collection']
        collection_path = os.path.join(_root, collection_name)
        all_paras.extend(TieBreaker(collection_path).gen_detailed_doc_stats_paras( 
            method_name,
            query_parts=q['qf_parts']
        ) )

    #print all_paras
    gen_batch_framework('gen_detailed_doc_stats', '11', all_paras, memory='10G')
  
def gen_detailed_doc_stats_atom(para_file):
    with open(para_file) as f:
        reader = csv.reader(f)
        for row in reader:
            collection_path = row[0]
            result_fn = row[1]
            output_fn = row[2]
            TieBreaker(collection_path).gen_doc_details_atom(result_fn, output_fn) 


def output_rel_tf_stats_batch():
    all_paras = []
    for q in g.query:
        collection_name = q['collection']
        collection_path = os.path.join(_root, collection_name)
        all_paras.append(RelTFStats(collection_path).batch_output_rel_tf_stats_paras())
    #print all_paras
    gen_batch_framework('output_rel_tf_stats', '32', all_paras)

def output_rel_tf_stats_atom(para_file):
    with open(para_file) as f:
        reader = csv.reader(f)
        for row in reader:
            collection_path = row[0]
            RelTFStats(collection_path).print_rel_tf_stats()


def gen_run_subqueries_batch(query_length=0):
    all_paras = []
    for q in g.query:
        collection_name = collection_name = q['collection_formal_name']
        collection_path = os.path.join(_root, q['collection'])
        all_paras.extend(RunSubqueries(collection_path, collection_name).batch_run_subqueries_paras(int(query_length)))
    #print all_paras
    gen_batch_framework('run_subqueries', '42', all_paras)

def run_subqueries_atom(para_file):
    with open(para_file) as f:
        reader = csv.reader(f)
        for row in reader:
            collection_path = row[0]
            collection_name = row[1]
            qid = row[2]
            query = row[3]
            subquery_id = row[4]
            indri_model_para = row[5]
            runfile_ofn = row[6]
            eval_ofn = row[7]
            RunSubqueries(collection_path, collection_name).run_subqueries(qid, subquery_id, query, indri_model_para, runfile_ofn, eval_ofn)

def gen_collect_subqueries_results_batch():
    all_paras = []
    for q in g.query:
        collection_name = collection_name = q['collection_formal_name']
        collection_path = os.path.join(_root, q['collection'])
        all_paras.extend(RunSubqueries(collection_path, collection_name).batch_collect_results_paras())
    gen_batch_framework('collect_subqueries_results', '44', all_paras)

def collect_subqueries_results_atom(para_file):
    with open(para_file) as f:
        reader = csv.reader(f)
        for row in reader:
            collection_path = row[0]
            collection_name = row[1]
            qid = row[2]
            RunSubqueries(collection_path, collection_name).collection_all_results(qid)

def output_subqueries_results(query_length):
    for q in g.query:
        collection_name = collection_name = q['collection_formal_name']
        collection_path = os.path.join(_root, q['collection'])
        RunSubqueries(collection_path, collection_name).output_results(int(query_length))

def output_optimal_dist():
    for q in g.query:
        collection_name = collection_name = q['collection_formal_name']
        collection_path = os.path.join(_root, q['collection'])
        RunSubqueries(collection_path, collection_name).output_optimal_dist()


def gen_subqueries_features_batch(feature_type):
    all_paras = []
    for q in g.query:
        collection_name = collection_name = q['collection_formal_name']
        collection_path = os.path.join(_root, q['collection'])
        all_paras.extend(SubqueriesLearning(collection_path, collection_name).batch_gen_subqueries_features_paras(int(feature_type)))
    #print all_paras
    gen_batch_framework('subqueries_features', '52', all_paras)

def gen_subqueries_features_atom(para_file):
    with open(para_file) as f:
        reader = csv.reader(f)
        for row in reader:
            collection_path = row[0]
            collection_name = row[1]
            qid = row[2]
            feature_type = row[3]
            SubqueriesLearning(collection_path, collection_name).gen_subqueries_features(qid, feature_type)


def output_features_kendallstau_batch(query_length):
    all_paras = []
    for q in g.query:
        collection_name = collection_name = q['collection_formal_name']
        collection_path = os.path.join(_root, q['collection'])
        all_paras.append((collection_path, collection_name, query_length))
    #print all_paras
    gen_batch_framework('output_features_kendallstau', '602', all_paras)

def output_features_kendallstau_atom(para_file):
    with open(para_file) as f:
        reader = csv.reader(f)
        for row in reader:
            collection_path = row[0]
            collection_name = row[1]
            query_length = int(row[2])
            SubqueriesLearning(collection_path, collection_name).output_features_kendallstau(query_length)

def output_features_kendallstau_all(query_length):
    query_length = int(query_length)
    all_paras = []
    for q in g.query:
        collection_name = collection_name = q['collection_formal_name']
        collection_path = os.path.join(_root, q['collection'])
        all_paras.append((collection_path, collection_name))
    SubqueriesLearning.output_features_kendallstau_all_collection(all_paras, query_length)

def output_features_selected_batch(query_length):
    all_paras = []
    for q in g.query:
        collection_name = collection_name = q['collection_formal_name']
        collection_path = os.path.join(_root, q['collection'])
        all_paras.append((collection_path, collection_name, query_length))
    #print all_paras
    gen_batch_framework('output_features_selected', '606', all_paras)

def output_features_selected_atom(para_file):
    with open(para_file) as f:
        reader = csv.reader(f)
        for row in reader:
            collection_path = row[0]
            collection_name = row[1]
            query_length = int(row[2])
            SubqueriesLearning(collection_path, collection_name).output_features_selected(query_length)

def output_features_selected_all(query_length):
    query_length = int(query_length)
    all_paras = []
    for q in g.query:
        collection_name = collection_name = q['collection_formal_name']
        collection_path = os.path.join(_root, q['collection'])
        all_paras.append((collection_path, collection_name))
    SubqueriesLearning.output_features_selected_all_collection(all_paras, query_length)

def output_features_classification_batch(query_length):
    all_paras = []
    for q in g.query:
        collection_name = collection_name = q['collection_formal_name']
        collection_path = os.path.join(_root, q['collection'])
        all_paras.append((collection_path, collection_name, query_length))
    #print all_paras
    gen_batch_framework('output_features_classification', '610', all_paras)

def output_features_classification_atom(para_file):
    with open(para_file) as f:
        reader = csv.reader(f)
        for row in reader:
            collection_path = row[0]
            collection_name = row[1]
            query_length = int(row[2])
            SubqueriesLearning(collection_path, collection_name).output_features_classification(query_length)

def cross_run_subquery_classification(query_length):
    query_length = int(query_length)
    collections = [(os.path.abspath(os.path.join(_root, q['collection'])), q['collection_formal_name']) for q in g.query]
    for i in range(len(collections)):
        this_training = []
        this_testing = []
        for j in range(len(collections)):
            if j == i:
                this_testing.append(collections[j])
            else:
                this_training.append(collections[j])
        SubqueriesLearning.cross_run_classification(this_training, this_testing, query_length)

def evaluate_cross_classification():
    collections = [(os.path.abspath(os.path.join(_root, q['collection'])), q['collection_formal_name']) for q in g.query]
    SubqueriesLearning.evaluate_cross_classification(collections)

def output_subqueries_features_batch(query_length):
    all_paras = []
    for q in g.query:
        collection_name = collection_name = q['collection_formal_name']
        collection_path = os.path.join(_root, q['collection'])
        all_paras.append((collection_path, collection_name, query_length))
    #print all_paras
    gen_batch_framework('output_subqueries_features', '62', all_paras)

def output_subqueries_features_atom(para_file):
    with open(para_file) as f:
        reader = csv.reader(f)
        for row in reader:
            collection_path = row[0]
            collection_name = row[1]
            query_length = int(row[2])
            SubqueriesLearning(collection_path, collection_name).output_collection_features(query_length)

def gen_svm_rank_batch(feature_type=1):
    all_paras = []
    for q in g.query:
        collection_name = collection_name = q['collection_formal_name']
        collection_path = os.path.join(_root, q['collection'])
        all_paras.extend(SubqueriesLearning(collection_path, collection_name).batch_gen_svm_rank_paras(feature_type))
    #print all_paras
    gen_batch_framework('svm_rank_train', '64', all_paras)

def svm_rank_atom(para_file):
    with open(para_file) as f:
        reader = csv.reader(f)
        for row in reader:
            collection_path = row[0]
            collection_name = row[1]
            folder = row[2]
            query_length = row[3]
            c = int(row[4])
            SubqueriesLearning(collection_path, collection_name).svm_rank_wrapper(folder, query_length, c)

def gen_evaluate_svm_model_batch(feature_type=1):
    all_paras = []
    for q in g.query:
        collection_name = collection_name = q['collection_formal_name']
        collection_path = os.path.join(_root, q['collection'])
        all_paras.append((collection_path, collection_name, feature_type))
    #print all_paras
    gen_batch_framework('evaluate_svm_rank_model', '66', all_paras)

def evaluate_svm_model_atom(para_file):
    with open(para_file) as f:
        reader = csv.reader(f)
        for row in reader:
            collection_path = row[0]
            collection_name = row[1]
            feature_type = int(row[2])
            SubqueriesLearning(collection_path, collection_name).evaluate_svm_model(feature_type)

def print_svm_model_feature_importance(feature_type=1, top=10):
    if feature_type == 1:
        folder = 'final'
    elif feature_type == 2:
        folder = 'kendallstau'
            
    all_top_features = {}
    for q in g.query:
        collection_path = os.path.join(_root, q['collection'])
        collection_name = q['collection_formal_name']
        all_top_features[collection_name] = {}
        res_folder = os.path.join(collection_path, 'subqueries', 'svm_rank', folder, 'featurerank')
        for fn in os.listdir(res_folder):
            query_length = int(fn)
            all_top_features[collection_name][query_length] = []
            with open(os.path.join(res_folder, fn)) as f:
                idx = 0
                for line in f:
                    line = line.strip()
                    if line:
                        all_top_features[collection_name][query_length].append(line.split(':')[0])
                    idx += 1
                    if idx >= top:
                        break
    print all_top_features
    print '### Top Features'
    print '| Query Len | 0 |'
    print '|--------|--------|--------|--------|'
    for collection_name in all_top_features:
        for idx in range(top):
            print '| **%s** | %s |' % (collection_name if idx == 0 else '', 
                all_top_features[collection_name][0][idx])

def cross_testing_svm_model(query_length=2):
    query_length = int(query_length)
    collections = [(os.path.abspath(os.path.join(_root, q['collection'])), q['collection_formal_name']) for q in g.query]
    for i in range(len(collections)):
        this_training = []
        this_testing = []
        for j in range(len(collections)):
            if j == i:
                this_testing.append(collections[j])
            else:
                this_training.append(collections[j])

        SubqueriesLearning.cross_testing(this_training, this_testing, query_length)

def evaluate_svm_cross_testing():
    collections = [(os.path.abspath(os.path.join(_root, q['collection'])), q['collection_formal_name']) for q in g.query]
    SubqueriesLearning.evaluate_svm_cross_testing(collections)


def mi_learn_batch(query_length, mi_distance, thres):
    all_paras = []
    for q in g.query:
        collection_name = collection_name = q['collection_formal_name']
        collection_path = os.path.join(_root, q['collection'])
        all_paras.append((collection_path, collection_name, query_length, mi_distance, thres))
    #print all_paras
    gen_batch_framework('mi_learn', 'mi_learn_atom', all_paras)

def mi_learn_atom(para_file):
    with open(para_file) as f:
        reader = csv.reader(f)
        for row in reader:
            collection_path = row[0]
            collection_name = row[1]
            query_length = int(row[2])
            mi_distance = int(row[3])
            thres = float(row[4])
            SubqueriesLearning(collection_path, collection_name).cluster_subqueries(query_length, mi_distance, thres)

def gen_resources_for_crowdsourcing():
    collections = [(os.path.abspath(os.path.join(_root, q['collection'])), q['collection_formal_name']) for q in g.query]
    SubqueriesLearning.gen_resources_for_crowdsourcing(collections)


###################################################
def run_all_baseline_results_atom(para_file):
    with open(para_file) as f:
        for line in f:
            line = line.strip()
            if line:
                row = line.split()
                indri.IndriRunQuery(row[0], row[2], row[1])


def run_all_baseline_results(collections=[]):
    frameinfo = getframeinfo(currentframe())
    current_function_name = inspect.stack()[0][3]
    result_dir = 'all_baseline_results'
    all_paras = []

    for c in collections:
        collection_name = c.split('/')[-1]
        if not collection_name:
            continue
        if not os.path.exists( os.path.join(c, result_dir) ):
            os.makedirs( os.path.join(c, result_dir) )
        q_path = os.path.join(os.path.abspath(c), 'standard_queries')
        for bs in np.arange(0., 1.01, 0.05):
            r_path = os.path.join(os.path.abspath(c), result_dir, 'pivoted_'+str(bs))
            if not os.path.exists(r_path):
                all_paras.append((q_path, '-rule=method:pivoted,s:%s' % bs, r_path))
            r_path = os.path.join(os.path.abspath(c), result_dir, 'okapi_'+str(bs))
            if not os.path.exists(r_path):
                all_paras.append((q_path, '-rule=method:okapi,b:%s' % bs, r_path))

        for miu in range(0,5001,250):
            r_path = os.path.join(os.path.abspath(c), result_dir, 'lm_'+str(miu))
            if not os.path.exists(r_path):
                all_paras.append((q_path, '-rule=method:d,mu:%s' % miu, r_path))

        if all_paras:
            #print all_paras
            MPI().gen_batch_framework(os.path.join(_root, collection_name, 'bin'), 
                current_function_name, frameinfo.filename, '21', 
                all_paras, 
                os.path.join(_root, collection_name, 'misc', current_function_name), 
                para_alreay_split=False,
                add_node_to_para=False,
                run_after_gen=True,
                memory='1G'
            )
        else:
            print 'Nothing to RUN for '+c

def gen_baseline_best_results(collections=[]):
    output_folder = 'baselines'
    if not os.path.exists( os.path.join(output_root, output_folder) ):
        os.makedirs( os.path.join(output_root, output_folder) )
    for c in collections:
        collection_name = c.split('/')[-1] if  c.split('/')[-1] else  c.split('/')[-2]
        MAP = {}
        for fn in os.listdir(os.path.join(c, 'all_baseline_results')):
            method = fn.split('_')[0] 
            evaluation = Evaluation(c, os.path.join(c, 'all_baseline_results', fn)).get_all_performance(return_all_metrics=True, metrics=['map'])
            for qid in evaluation:
                if qid not in MAP:
                    MAP[qid] = {}
                if method not in MAP[qid]:
                    MAP[qid][method] = {}
                MAP[qid][method][fn] = evaluation[qid]['map']
        MAP1 = {}
        for qid in MAP:
            MAP1[qid] = []
            for method in MAP[qid]:
                MAP1[qid].append(sorted(MAP[qid][method].items(), key=itemgetter(1), reverse=True)[0])

        with open(os.path.join(output_root, output_folder, collection_name+'-all_baseline_results.csv'), 'wb') as output:
            for qid in sorted(MAP1.keys()):
                output.write('%s' % qid)
                for ele in MAP1[qid]:
                    output.write(',%s,%s' % (ele[0], ele[1]))
                output.write('\n')


def gen_run_baseline_paras(baseline_method='lm'):
    all_paras = []
    with open('g.json') as f:
        methods = json.load(f)['methods']
        for m in methods:
            for q in g.query:
                collection_name = q['collection']
                collection_path = os.path.join(_root, collection_name)
                all_paras.extend(TieBreaker(collection_path).gen_run_baseline_paras(m['name'], 10000, q['qf_parts']))
    gen_batch_framework('run_baseline', '00', all_paras)

def gen_run_baseline_atom(para_file):
    with open(para_file) as f:
        reader = csv.reader(f)
        for row in reader:
            query_para = row[0]
            output_fn = row[1]
            #print query_para
            #print output_fn
            run_query(query_para, output_fn)
            
def run_query(query_para, output_fn):
    paras = shlex.split(query_para)
    paras.insert(0, 'IndriRunQuery_EX')
    p = Popen(paras, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p.communicate()
    if 'exiting' not in stdout:
        with open(output_fn, 'wb') as o:
            o.write(stdout)
    else:
        print stdout, stderr
        exit()   




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-0", "--gen_run_baseline_paras",
        action='store_true',
        help="generate the batch run parameters for running \
               the baseline method")
    parser.add_argument('-00', '--gen_run_baseline_atom', nargs=1,
                       help='actually run the baseline method')

    parser.add_argument("-1", "--gen_detailed_doc_stats_paras",
        nargs=1,
        help="Generate the detailed document for the query. The input is \
            the method name (default lm).")
    parser.add_argument("-11", "--gen_detailed_doc_stats_atom",
        nargs=1,
        help="Generate the detailed document for the query. The input is \
            the result file.")


    parser.add_argument('-31', '--output_rel_tf_stats_batch', 
        action='store_true',
        help='Output the term frequency of query terms for relevant documents.')
    parser.add_argument('-32', '--output_rel_tf_stats_atom', 
        nargs=1,
        help='Output the term frequency of query terms for relevant documents.')

    parser.add_argument('-41', '--gen_run_subqueries_batch', 
        nargs=1,
        help='generate run subqueries paras. para indicating the query length')
    parser.add_argument('-42', '--run_subqueries_atom', 
        nargs=1,
        help='generate run subqueries paras')
    parser.add_argument('-43', '--gen_collect_subqueries_results_batch', 
        action='store_true',
        help='collect all the results')
    parser.add_argument('-44', '--collect_subqueries_results_atom', 
        nargs=1,
        help='generate run subqueries paras')
    parser.add_argument('-45', '--output_subqueries_results', 
        nargs=1,
        help='arg: query_length')
    parser.add_argument('-46', '--output_optimal_dist', 
        action='store_true',
        help='output the optimal performances distribution')


    parser.add_argument('-51', '--gen_subqueries_features_batch', 
        nargs=1, # feature type
        help='generate subqueries features paras.')
    parser.add_argument('-52', '--gen_subqueries_features_atom', 
        nargs=1,
        help='generate subqueries features')
    parser.add_argument('-601', '--output_features_kendallstau_batch', 
        nargs=1,
        help='generate features kendallstau with performance. paras. arg: query length (0 for all queries)')
    parser.add_argument('-602', '--output_features_kendallstau_atom', 
        nargs=1,
        help='generate features kendallstau with performance')
    parser.add_argument('-603', '--output_features_kendallstau_all', 
        nargs=1,
        help='generate kendallstau features for all collections. arg: query length')

    parser.add_argument('-605', '--output_features_selected_batch', 
        nargs=1,
        help='generate features selected with performance. paras. arg: query length (0 for all queries)')
    parser.add_argument('-606', '--output_features_selected_atom', 
        nargs=1,
        help='generate features selected with performance')
    parser.add_argument('-607', '--output_features_selected_all', 
        nargs=1,
        help='generate selected features for all collections. arg: query length')

    parser.add_argument('-609', '--output_features_classification_batch', 
        nargs=1,
        help='generate features classification with performance. paras. arg: query length (0 for all queries)')
    parser.add_argument('-610', '--output_features_classification_atom', 
        nargs=1,
        help='generate features classification with performance')
    parser.add_argument('-611', '--cross_run_subquery_classification', 
        nargs=1,
        help=('run classification on the original queries to see whether '
            'it should favor subquery or the original one. '
            'arg: query_length')
        )
    parser.add_argument('-612', '--evaluate_cross_classification', 
        action='store_true',
        help='evaluate_cross_classification')
    parser.add_argument('-61', '--output_subqueries_features_batch', 
        nargs=1,
        help='generate subqueries features paras. arg: query length (0 for all queries)')
    parser.add_argument('-62', '--output_subqueries_features_atom', 
        nargs=1,
        help='generate subqueries features')
    parser.add_argument('-63', '--gen_svm_rank_batch', 
        nargs=1,
        help=('generate the batch runs for svm rank. '
            'arg: [feature_type(1-all features, 2-top features gen by kendallstau correlation)]')
    )
    parser.add_argument('-64', '--svm_rank_atom', 
        nargs=1,
        help='svm rank atom')
    parser.add_argument('-65', '--gen_evaluate_svm_model_batch', 
        nargs=1,
        help='generate the batch runs for svm rank. arg: [feature_type(1-all features, 2-top features gen by kendallstau correlation')
    parser.add_argument('-66', '--evaluate_svm_model_atom', 
        nargs=1,
        help='svm rank atom')
    parser.add_argument('-67', '--print_svm_model_feature_importance', 
        nargs=2,
        help=('print the top features of svm model.'
         ' arg: [feature_type(1-all features, 2-top features gen by kendallstau correlation)] '
         '[N (top N will be printed)]'))
    parser.add_argument('-68', '--svm_cross_testing', 
        nargs=1,
        help='cross testing the svm rank. arg: query_length')
    parser.add_argument('-69', '--evaluate_svm_cross_testing', 
        action='store_true',
        help='evaluate cross testing the svm rank')

    parser.add_argument('-mi_learn_batch', '--mi_learn_batch', 
        nargs=3,
        help='arg: [query_length (0 for all queries)] [mi_distance] [cluster_threshold]')
    parser.add_argument('-mi_learn_atom', '--mi_learn_atom', 
        nargs=1,
        help='learn the subquery performances using only Mutual Information')

    parser.add_argument('-gen_crowdsourcing', '--gen_crowdsourcing', 
        action='store_true',
        help='')

    parser.add_argument("-2", "--run_all_baseline_results",
        nargs='+',
        help="Run all parameters for Pivoted, Okapi and Dirichlet.")

    parser.add_argument("-21", "--run_all_baseline_results_atom",
        nargs=1,
        help="Run all parameters for Pivoted, Okapi and Dirichlet. This actually does the work.")

    parser.add_argument("-22", "--gen_baseline_best_results",
        nargs='+',
        help="Output the baseline best results.")



    args = parser.parse_args()

    if args.gen_run_baseline_paras:
        gen_run_baseline_paras()
    if args.gen_run_baseline_atom:
        gen_run_baseline_atom(args.gen_run_baseline_atom[0])

    if args.gen_detailed_doc_stats_paras:
        gen_detailed_doc_stats_paras(args.gen_detailed_doc_stats_paras[0])
    if args.gen_detailed_doc_stats_atom:
        gen_detailed_doc_stats_atom(args.gen_detailed_doc_stats_atom[0])

    if args.output_rel_tf_stats_batch:
        output_rel_tf_stats_batch()
    if args.output_rel_tf_stats_atom:
        output_rel_tf_stats_atom(args.output_rel_tf_stats_atom[0])

    if args.gen_run_subqueries_batch:
        gen_run_subqueries_batch(args.gen_run_subqueries_batch[0])
    if args.run_subqueries_atom:
        run_subqueries_atom(args.run_subqueries_atom[0])
    if args.gen_collect_subqueries_results_batch:
        gen_collect_subqueries_results_batch()
    if args.collect_subqueries_results_atom:
        collect_subqueries_results_atom(args.collect_subqueries_results_atom[0])
    if args.output_subqueries_results:
        output_subqueries_results(args.output_subqueries_results[0])
    if args.output_optimal_dist:
        output_optimal_dist()

    if args.gen_subqueries_features_batch:
        gen_subqueries_features_batch(args.gen_subqueries_features_batch[0])
    if args.gen_subqueries_features_atom:
        gen_subqueries_features_atom(args.gen_subqueries_features_atom[0])
    if args.output_features_kendallstau_batch:
        output_features_kendallstau_batch(args.output_features_kendallstau_batch[0])
    if args.output_features_kendallstau_atom:
        output_features_kendallstau_atom(args.output_features_kendallstau_atom[0])
    if args.output_features_kendallstau_all:
        output_features_kendallstau_all(args.output_features_kendallstau_all[0])
    if args.output_features_selected_batch:
        output_features_selected_batch(args.output_features_selected_batch[0])
    if args.output_features_selected_atom:
        output_features_selected_atom(args.output_features_selected_atom[0])
    if args.output_features_selected_all:
        output_features_selected_all(args.output_features_selected_all[0])
    if args.output_features_classification_batch:
        output_features_classification_batch(args.output_features_classification_batch[0])
    if args.output_features_classification_atom:
        output_features_classification_atom(args.output_features_classification_atom[0])
    if args.cross_run_subquery_classification:
        cross_run_subquery_classification(args.cross_run_subquery_classification[0])
    if args.evaluate_cross_classification:
        evaluate_cross_classification()
    if args.output_subqueries_features_batch:
        output_subqueries_features_batch(args.output_subqueries_features_batch[0])
    if args.output_subqueries_features_atom:
        output_subqueries_features_atom(args.output_subqueries_features_atom[0])
    if args.gen_svm_rank_batch:
        gen_svm_rank_batch(int(args.gen_svm_rank_batch[0]))
    if args.svm_rank_atom:
        svm_rank_atom(args.svm_rank_atom[0])
    if args.gen_evaluate_svm_model_batch:
        gen_evaluate_svm_model_batch(int(args.gen_evaluate_svm_model_batch[0]))
    if args.evaluate_svm_model_atom:
        evaluate_svm_model_atom(args.evaluate_svm_model_atom[0])
    if args.print_svm_model_feature_importance:
        print_svm_model_feature_importance(int(args.print_svm_model_feature_importance[0]))
    if args.svm_cross_testing:
        cross_testing_svm_model(int(args.svm_cross_testing[0]))
    if args.evaluate_svm_cross_testing:
        evaluate_svm_cross_testing()

    if args.mi_learn_batch:
        mi_learn_batch(args.mi_learn_batch[0], args.mi_learn_batch[1], args.mi_learn_batch[2])
    if args.mi_learn_atom:
        mi_learn_atom(args.mi_learn_atom[0])

    if args.gen_crowdsourcing:
        gen_resources_for_crowdsourcing()

    if args.run_all_baseline_results:
        run_all_baseline_results(args.run_all_baseline_results)

    if args.run_all_baseline_results_atom:
        run_all_baseline_results_atom(args.run_all_baseline_results_atom[0])

    if args.gen_baseline_best_results:
        gen_baseline_best_results(args.gen_baseline_best_results)    

