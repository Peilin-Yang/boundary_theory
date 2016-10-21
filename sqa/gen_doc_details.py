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

class GenSqaDocDetails(object):
    def __init__(self, path):
        self.collection_path = os.path.abspath(path)
        if not os.path.exists(self.collection_path):
            print '[CollectionStats Constructor]:Please provide a valid collection path'
            exit(1)

        self.dumpindex = 'dumpindex_EX'
        self.index_path = os.path.join(self.collection_path, 'index')
        self.queries = {ele['num']:ele['title'] for ele in Query(self.collection_path).get_queries_of_length(1)}
        print self.queries
        self.judgements = Judgment(self.collection_path)\
            .get_judgment_of_some_queries(self.queries.keys(), 'dict')

    def batch_gen_doc_details_paras(self):
        paras = []
        output_root = os.path.join(self.collection_path, 'sqa_doc_details')
        for qid in self.queries:
            if not os.path.exists(os.path.join(output_root, qid)):
                paras.append((self.collection_path, qid))
        return paras

    def output_doc_details(self, qid):
        process = Popen([self.dumpindex, self.index_path, 't', self.queries[qid]['title']], stdout=PIPE)
        stdout, stderr = process.communicate()
        details = []
        for line in stdout.split('\n')[1:-1]:
            line = line.strip()
            if line:
                row = line.split()
                docid = row[1]
                tf = int(row[2])
                doc_len = int(row[3])
                rel_score = self.judgements[qid][docid] if docid in self.judgements[qid] else 0
                details.append({
                    'qid': qid,
                    'docid': docid,
                    'rel_score': rel_score,
                    'total_tf': tf,
                    'doc_len': doc_len
                })

        ofn = os.path.join(self.collection_path, 'sqa_doc_details', qid)
        with open(ofn, 'wb') as f:
            fieldnames = ['qid','docid','rel_score','total_tf','doc_len']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(details)



if __name__ == '__main__':
    GenSqaDocDetails(sys.argv[1]).output_doc_details(sys.argv[2])
