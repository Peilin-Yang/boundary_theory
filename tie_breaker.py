import sys,os
import math
import argparse
import json
import ast
from datetime import datetime
from operator import itemgetter
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

    def get_tf_docln_stats(self, term, all_docnos):
        r = {}
        process = Popen(['dumpindex', os.path.join(self.collection_path, 'index'), 't', term], stdout=PIPE)
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

    def gen_doc_details(self, rfn):
        results = ResultsFile(rfn).get_all_results('list')
        return results


    def gen_doc_details_atom(self, para_fn):
        with open(para_fn) as f:
            all_paras = [line.split() for line in f.readlines() if line.strip()]
        all_qids = set([ele[1] for ele in all_paras])
        all_docnos = set([ele[2] for ele in all_paras])
        queries = Query(self.collection_path).get_queries_dict()
        judgement = Judgment(self.collection_path).get_all_judgments('dict')
        #print queries
        #print judgement
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

        docs_stats = CollectionStats(all_paras[0][0]).get_document_stats(all_internal_doc_ids)

        output_fn = all_paras[0][-1]
        for para in all_paras:
            qid = para[1]
            doc_id = para[2]
            internal_doc_id = all_docids_mapping[doc_id]
            score = para[3]
            with open(output_fn, 'ab') as output:
                output.write('%s,%s,%s,' % (qid, doc_id, score))
                if doc_id in judgement[qid]:
                    output.write('%d' % judgement[qid][doc_id])
                else:
                    output.write('0')

                output.write(',"')
                idx = 1
                doc_length = 0
                total_tf = 0
                for query_term in doc_term_stats[qid]:
                    if doc_id in doc_term_stats[qid][query_term]:
                        total_tf += int(doc_term_stats[qid][query_term][doc_id][0])
                        doc_length = doc_term_stats[qid][query_term][doc_id][1]
                        if idx != 1:
                            output.write(',')
                        output.write('%s-%s' % (query_term, \
                            doc_term_stats[qid][query_term][doc_id][0]))
                        idx += 1
                output.write('"')
                output.write(',%s' % str(total_tf))
                output.write(',%s' % doc_length)
                output.write(',%.4f,%.4f,%.4f,%.4f\n' % 
                    (
                    docs_stats[internal_doc_id]['minTF'], 
                    docs_stats[internal_doc_id]['maxTF'], 
                    docs_stats[internal_doc_id]['avgTF'], 
                    docs_stats[internal_doc_id]['varTF'], 
                    ) 
                )

                
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-1', '--gen_doc_details', nargs=2,
                       help='generate the TF, IDF, DOC Length and other detailed \
                       information for documents. The input is the ranking list \
                       file by runing IndriRunQuery and the method using the default \
                       language model method. The results are limited to 1000 documents. \
                       NOTICE: This step only split the result file to pieces so that \
                       MPI program can run all the docs in parallel!')
    
    parser.add_argument('-11', '--gen_doc_details_atom', nargs=1,
                       help='generate the TF, IDF, DOC Length and other detailed \
                       information for documents. The input is the ranking list \
                       file by runing IndriRunQuery and the method using the default \
                       language model method. The results are limited to 1000 documents. \
                       NOTICE: This step actually do the work! \
                       The input contains all the parameters. We want one query processed \
                       by one computing node and one computing node can process multiple \
                       queries.')


    args = parser.parse_args()

    if args.gen_doc_details:
        TieBreaker(args.gen_doc_details[0]).gen_doc_details(args.gen_doc_details[1])
    if args.gen_doc_details_atom:
        with open(args.gen_doc_details_atom[0]) as f:
            first_line = f.readline()
            collection_path = first_line.split()[0]
            TieBreaker(collection_path).gen_doc_details_atom(args.gen_doc_details_atom[0])
  

