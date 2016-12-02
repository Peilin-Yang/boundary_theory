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
from collection_stats import CollectionStats

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

    def get_data(self, qids=[]):
        all_data = []
        for qid in qids:
            # try:
            with open(os.path.join(self.data_root, qid)) as f:
                data = json.load(f)
                all_data[qid] = data
            # except:
            #     print 'Loading ' + qid +' data failed!'
        return all_data



if __name__ == '__main__':
    RelTFStats(sys.argv[1]).get_data(sys.argv[2])
