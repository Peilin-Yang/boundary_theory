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
        self.queries = {ele['num']:ele['title'] for ele in Query(self.collection_path).get_queries_of_length(1)}
        self.judgements = Judgment(self.collection_path)\
            .get_judgment_of_some_queries(self.queries.keys(), 'dict')
        self.doc_details_root = os.path.join(self.collection_path, 'doc_details')
        if not os.path.exists(self.doc_details_root):
            os.makedirs(self.doc_details_root)

    def get_qid_details(self, qid):
        with open(os.path.join(self.doc_details_root, qid)) as f:
            rows = csv.DictReader(f)
            for row in rows:
                yield row

if __name__ == '__main__':
    GenSqaDocDetails(sys.argv[1]).output_doc_details(sys.argv[2])
