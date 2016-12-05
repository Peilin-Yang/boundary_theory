import os,sys
import csv
import ast
import json
import tempfile
import subprocess
from subprocess import Popen, PIPE

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils/'))
from query import Query
from judgment import Judgment
from performance import Performances
from collection_stats import CollectionStats
from gen_doc_details import GenDocDetails

import numpy as np

class RelTFStats(object):
    """
    the tf stats of each query term in the relevant docs
    """
    def __init__(self, path):
        self.collection_path = os.path.abspath(path)
        if not os.path.exists(self.collection_path):
            print '[CollectionStats Constructor]:Please provide a valid collection path'
            exit(1)

        self.data_root = os.path.join(self.collection_path, 'rel_tf_stats')

    def batch_output_rel_tf_stats_paras(self):
        """
        Output the term frequency of query terms for relevant documents.
        For example, a query {journalist risks} will output 
        {
            'journalist': {'mean': 6.8, 'std': 1.0}, the average TF for journalist in relevant docs is 6.8
            'risks': {'mean': 1.6, 'std': 5.0}
        }
        """
        paras = []
        output_root = os.path.join(self.collection_path, 'rel_tf_stats')
        if not os.path.exists(output_root):
            os.makedirs(output_root)
        paras.append((self.collection_path))
        return paras

    def print_rel_tf_stats(self):
        queries = Query(self.collection_path).get_queries()
        queries = {ele['num']:ele['title'] for ele in queries}
        cs = CollectionStats(self.collection_path)
        doc_details = GenDocDetails(self.collection_path)
        rel_cnts = {qid:len(rel_docs_list) for qid, rel_docs_list in Judgment(self.collection_path).get_relevant_docs_of_some_queries(queries.keys()).items()}
        for qid in queries:
            #if not os.path.exists(os.path.join(output_root, qid)):
            terms, tfs, dfs, doclens = doc_details.get_only_rels(qid)
            tf_mean = np.mean(tfs, axis=1)
            tf_std = np.std(tfs, axis=1)
            idfs = np.log((cs.get_doc_counts() + 1)/(dfs+1e-4))
            try:
                okapi_perform = Performances(self.collection_path).gen_optimal_performances_queries('okapi', [qid])[0]
                lm_perform = Performances(self.collection_path).gen_optimal_performances_queries('dir', [qid])[0]
                terms_stats = {
                    t:{'mean': tf_mean[idx], 'std': tf_std[idx], 
                    'df': dfs[idx], 'idf': idfs[idx], 'tfc':cs.get_term_collection_occur(t), 
                    'ptC': cs.get_term_collection_occur(t)*1./cs.get_total_terms(),
                    'zero_cnt_percentage': round(1.0-np.count_nonzero(tfs[idx])*1./tfs[idx].size, 2)
                } for idx, t in enumerate(terms) if dfs[idx] != 0}
            except:
                continue
            output = {
                'AP': {'okapi': okapi_perform, 'dir': lm_perform},
                'rel_cnt': rel_cnts[qid],
                'terms': terms_stats
            }
            output_root = os.path.join(self.collection_path, 'rel_tf_stats')
            with open(os.path.join(output_root, qid), 'w') as f:
                json.dump(output, f, indent=2)

    def get_data(self, qids=[]):
        all_data = {}
        for qid in qids:
            try:
                with open(os.path.join(self.data_root, qid)) as f:
                    data = json.load(f)
                    all_data[qid] = data
            except:
                print 'Loading ' + qid +' data failed!'
        return all_data



if __name__ == '__main__':
    RelTFStats(sys.argv[1]).get_data(sys.argv[2])
