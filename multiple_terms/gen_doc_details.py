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

class GenDocDetails(object):
    def __init__(self, path):
        self.collection_path = os.path.abspath(path)
        if not os.path.exists(self.collection_path):
            print '[CollectionStats Constructor]:Please provide a valid collection path'
            exit(1)

        self.dumpindex = 'dumpindex_EX'
        self.index_path = os.path.join(self.collection_path, 'index')
        self.queries = {ele['num']:ele['title'] \
            for ele in Query(self.collection_path).get_queries()}
        self.judgements = Judgment(self.collection_path)\
            .get_judgment_of_some_queries(self.queries.keys(), 'dict')
        self.doc_details_root = os.path.join(self.collection_path, 'doc_details')
        if not os.path.exists(self.doc_details_root):
            os.makedirs(self.doc_details_root)

        self.baseline_fn = os.path.join(self.collection_path, 'baselines', 'lm_10000')

    def batch_gen_doc_details_paras(self):
        paras = []
        output_root = self.doc_details_root
        for qid, query in self.queries.items():
            if not os.path.exists(os.path.join(output_root, qid)):
                paras.append((self.collection_path, qid, query))
        return paras

    def read_docid_from_baseline(self, req_qid):
        docids = {}
        with open(self.baseline_fn) as f:
            for line in f:
                line = line.strip()
                if line:
                    row = line.split()
                    qid = row[0]
                    if qid != req_qid:
                        continue
                    docid = row[2]
                    score = float(row[4])
                    docids[docid] = score
        return docids

    def output_doc_details(self, qid, query):
        if not self.judgements[qid]:
            print self.collection_path, qid
            return
        candidate_docids = self.read_docid_from_baseline(qid)
        terms_list = query.split()
        terms_set = set(terms_list)
        docs = {}
        for t in terms_set:
            process = Popen([self.dumpindex, self.index_path, 't', t], stdout=PIPE)
            stdout, stderr = process.communicate()
            for line in stdout.split('\n')[1:-2]:
                line = line.strip()
                if line:
                    row = line.split()
                    docid = row[1]
                    if docid not in candidate_docids:
                        continue
                    tf = int(row[2])
                    doc_len = int(row[3])
                    rel_score = self.judgements[qid][docid] if docid in self.judgements[qid] else 0
                    if docid not in docs:
                        docs[docid] = {}
                    if t not in docs[docid]:
                        docs[docid][t] = 0
                    if 'total_tf' not in docs[docid]:
                        docs[docid]['total_tf'] = 0
                    if 'doc_len' not in docs[docid]:
                        docs[docid]['doc_len'] = doc_len
                    if 'rel_score' not in docs[docid]:
                        docs[docid]['rel_score'] = rel_score
                    docs[docid][t] += tf
                    docs[docid]['total_tf'] += tf

        ofn = os.path.join(self.doc_details_root, qid)
        with open(ofn, 'wb') as f:
            fieldnames = ['qid','docid','rel_score','tf','total_tf','doc_len']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for docid in docs:
                this_doc = {
                    'qid': qid,
                    'docid': docid,
                    'rel_score': docs[docid]['rel_score'],
                    'total_tf': docs[docid]['total_tf'],
                    'doc_len': docs[docid]['doc_len']
                }
                tf = []
                for t in terms_list:
                    tf.append(t+'-'+str(docs[docid][t] if t in docs[docid] else 0))
                this_doc['tf'] = ','.join(tf)
                writer.writerow(this_doc)

    def get_qid_details(self, qid):
        with open(os.path.join(self.doc_details_root, qid)) as f:
            rows = csv.DictReader(f)
            for row in rows:
                yield row

if __name__ == '__main__':
    GenDocDetails(sys.argv[1]).output_doc_details(sys.argv[2])
