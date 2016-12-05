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
    gen_batch_framework('output_rel_tf_stats', '22', all_paras)

def output_rel_tf_stats_atom(para_file):
    with open(para_file) as f:
        reader = csv.reader(f)
        for row in reader:
            collection_path = row[0]
            RelTFStats(collection_path).print_rel_tf_stats()


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

    if args.run_all_baseline_results:
        run_all_baseline_results(args.run_all_baseline_results)

    if args.run_all_baseline_results_atom:
        run_all_baseline_results_atom(args.run_all_baseline_results_atom[0])

    if args.gen_baseline_best_results:
        gen_baseline_best_results(args.gen_baseline_best_results)    

