import sys, os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils/'))
from query import Query
from results_file import ResultsFile
from judgment import Judgment
from evaluation import Evaluation
from utils import Utils
from collection_stats import CollectionStats
from baselines import Baselines
import g
import ArrayJob

class PerformaceAnalysis(object):
    """
    Performance Analysis related
    """
    def __init__(self):
        pass

    def gen_performance_trending(self, collection='../../../reproduce/collections/wt2g'):
        """
        Generate the performance trendings based on the parameter of 
        the model and the number of terms in the query
        @Input:
            - 

        @Output: csv formatted
        """
        with open('g.json') as f:
            models = [{ele['name']:ele['paras']} for ele in json.load(f)['methods']]
        query_instance = Query(collection)
        eval_instance = Evaluation(collection)
        for i in query_instance.get_queries_lengths():
            qids = [q['num'] for q in query_instance.get_queries_of_length(i)]
            for model in models:
                for para_key, para_value in model.items():
                    avg_perform = []
                    x = [(para-min(para_value.values()[0]))/(max(para_value.values()[0]) - min(para_value.values()[0])) for para in para_value.values()[0]]
                    for para in para_value.values()[0]:
                        method_str = para_key+','+para_value.iterkeys().next()+':'+str(para)
                        avg_peform.append( np.mean([v['map'] for v in eval_instance.get_all_performance_of_some_queries(method = method_str, qids = qids, return_all_metrics = False).values()]) )
                    print x, avg_perform

PerformaceAnalysis().gen_performance_trending()

