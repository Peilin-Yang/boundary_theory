import sys,os
import math
import argparse
import json
import ast
import xml.etree.ElementTree as ET
import csv
import re
from subprocess import Popen, PIPE

from utils.query import Query
from utils.results_file import ResultsFile
from utils.judgment import Judgment
from utils.evaluation import Evaluation
from utils.utils import Utils
from utils.collection_stats import CollectionStats
from utils.evaluation import Evaluation

from inspect import currentframe, getframeinfo
import numpy as np


class TieBreaker(object):
    def __init__(self, path):
        self.collection_path = os.path.abspath(path)
        if not os.path.exists(self.collection_path):
            frameinfo = getframeinfo(currentframe())
            print frameinfo.filename, frameinfo.lineno
            print '[TieBreaker Constructor]:Please provide a valid collection path'
            exit(1)

        self.fieldnames=['qid', 'docid', 'score', 'rel_score', 'tf', 'total_tf', 
            'doc_len', 'doc_minTF', 'doc_maxTF', 'doc_avgTF', 'doc_varTF']
        self.detailed_doc_stats_folder = os.path.join(self.collection_path, 'detailed_doc_stats')
        if not os.path.exists(self.detailed_doc_stats_folder):
            os.makedirs(self.detailed_doc_stats_folder)

    def get_tf_docln_stats(self, term, all_docnos):
        r = {}
        process = Popen(['dumpindex_EX', os.path.join(self.collection_path, 'index'), 't', term], stdout=PIPE)
        stdout, stderr = process.communicate()
        #print stdout
        all_internal_doc_ids = []
        docids_mapping = {}

        for line in stdout.split('\n')[1:-2]: 
            #first line shows term's stem, total count in collection, total terms in collection
            #last second line shows total docs # which contain this term
            #last line is '\n'
            line = line.strip()
            if line:
                row = line.split()
                internal_doc_id = row[0]
                external_docno = row[1]
                docids_mapping[external_docno] = internal_doc_id
                if external_docno in all_docnos:
                    all_internal_doc_ids.append(internal_doc_id)
                tf = row[2]
                doc_length = row[3]
                r[external_docno] = ((tf, doc_length))
                #print term, tf, doc_length

        #print term, r
        #raw_input()
        return r, all_internal_doc_ids, docids_mapping

    def gen_run_baseline_paras(self, baseline_method='', _count=10000, query_parts=['title']):
        """
        Generate the batch run parameters for running baseline method.
        This is essentially generating the IndriRunQuery_EX command line 
        to run all the queries for the baseline method. 
        count is the number of results returned for each query. We set 
        the default value as 10,000
        """
        all_queries = Query(self.collection_path).get_queries()
        all_paras = []
        for qp in query_parts:
            for q in all_queries:
                query_qid = q['num']
                output_fn = os.path.join(self.collection_path, 'split_results', qp+'_'+query_qid+'-method:%s' % baseline_method)
                if os.path.exists(output_fn):
                    continue
                index = os.path.join(self.collection_path, 'index')
                trec_format = 'true'
                count = _count
                query_text = q[qp]
                query_str_list = [
                    '-index=%s' % index, 
                    '-count=%d' % count, 
                    '-trecFormat=%s' % trec_format,
                    '-query.number=%s' % query_qid,
                    '-query.text="%s"' % query_text,  
                    '-method=%s' % baseline_method, 
                ]
                query_str = ' '.join(query_str_list)
                all_paras.append((query_str, output_fn))
        return all_paras

    def gen_detailed_doc_stats_paras(self, method_name, query_parts=['title']):
        all_queries = Query(self.collection_path).get_queries()
        all_paras = []
        for qp in query_parts:
            for q in all_queries:
                query_qid = q['num']
                input_fn = os.path.join(self.collection_path, 'split_results', qp+'_'+query_qid+'-method:%s' % method_name)
                output_fn = os.path.join(self.detailed_doc_stats_folder, query_qid)
                if os.path.exists(output_fn):
                    continue
                all_paras.append((self.collection_path, input_fn, output_fn))
        return all_paras

    def gen_doc_details_atom(self, input_fn, output_fn):
        with open(input_fn) as f:
            all_paras = [line.split() for line in f.readlines() if line.strip()]
        all_qids = set([ele[0] for ele in all_paras])
        all_docnos = set([ele[2] for ele in all_paras])
        queries = Query(self.collection_path).get_queries_dict()
        judgements = Judgment(self.collection_path).get_all_judgments('dict')
        #print queries
        #print judgements
        doc_term_stats = {}
        all_docids_mapping = {}
        all_internal_doc_ids = []
        for qid in all_qids:
            query_terms = queries[qid]['title'].split()
            doc_term_stats[qid] = {}
            for query_term in query_terms:
                doc_term_stats[qid][query_term], internal_doc_ids, docids_mapping \
                    = self.get_tf_docln_stats(query_term, all_docnos)
                all_docids_mapping.update(docids_mapping)
                all_internal_doc_ids.extend(internal_doc_ids)

        docs_stats = CollectionStats(self.collection_path).get_document_stats(all_internal_doc_ids)

        all_docs = []
        for para in all_paras:
            qid = para[0]
            doc_id = para[2]
            internal_doc_id = all_docids_mapping[doc_id]
            score = para[4]
            if qid not in judgements:
                continue
            judgment = judgements[qid][doc_id] if doc_id in judgements[qid] else 0
            doc_length = 0
            total_tf = 0
            all_tfs = []
            for query_term in doc_term_stats[qid]:
                if doc_id in doc_term_stats[qid][query_term]:
                    total_tf += int(doc_term_stats[qid][query_term][doc_id][0])
                    doc_length = doc_term_stats[qid][query_term][doc_id][1]
                    all_tfs.append('%s-%s' % (query_term, doc_term_stats[qid][query_term][doc_id][0]))
            this_obj = {
                'qid': qid,
                'docid': doc_id,
                'score': score,
                'rel_score': judgment,
                'tf': ','.join(all_tfs),
                'total_tf': total_tf,
                'doc_len': doc_length,
                'doc_minTF': docs_stats[internal_doc_id]['minTF'], 
                'doc_maxTF': docs_stats[internal_doc_id]['maxTF'], 
                'doc_avgTF': docs_stats[internal_doc_id]['avgTF'], 
                'doc_varTF': docs_stats[internal_doc_id]['varTF']
            }
            all_docs.append(this_obj)
        with open(output_fn, 'wb') as output:
            writer = csv.DictWriter(output, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(all_docs)
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-1', '--gen_run_baseline_paras', nargs='+',
                       help='generate the batch run parameters for running \
                       the baseline method')
    
    parser.add_argument('-11', '--gen_run_baseline_atom', nargs=1,
                       help='actually run the baseline method')

    parser.add_argument('-2', '--gen_doc_details', nargs=2,
                       help='generate the TF, IDF, DOC Length and other detailed \
                       information for documents. The input is the ranking list \
                       file by runing IndriRunQuery and the method using the default \
                       language model method. The results are limited to 1000 documents. \
                       NOTICE: This step only split the result file to pieces so that \
                       MPI program can run all the docs in parallel!')
    
    parser.add_argument('-22', '--gen_doc_details_atom', nargs=1,
                       help='generate the TF, IDF, DOC Length and other detailed \
                       information for documents. The input is the ranking list \
                       file by runing IndriRunQuery and the method using the default \
                       language model method. The results are limited to 1000 documents. \
                       NOTICE: This step actually do the work! \
                       The input contains all the parameters. We want one query processed \
                       by one computing node and one computing node can process multiple \
                       queries.')


    args = parser.parse_args()

    if args.gen_run_baseline_paras:
        TieBreaker(args.gen_run_baseline_paras[0]).gen_run_baseline_paras(*args.gen_run_baseline_paras[1:])
    if args.gen_run_baseline_atom:
        gen_run_baseline_atom(args.gen_run_baseline_atom[0])

    if args.gen_doc_details:
        TieBreaker(args.gen_doc_details[0]).gen_doc_details(args.gen_doc_details[1])
    if args.gen_doc_details_atom:
        with open(args.gen_doc_details_atom[0]) as f:
            first_line = f.readline()
            collection_path = first_line.split()[0]
            TieBreaker(collection_path).gen_doc_details_atom(args.gen_doc_details_atom[0])
  

