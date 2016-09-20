import sys, os
import json
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

    def gen_performance_trending(self, collection='../../../reproduce/collections/wt2g', method='okapi', outputformat='png'):
        """
        Generate the performance trendings based on the parameter of 
        the model and the number of terms in the query
        @Input:
            - qids (list) : a list contains the qid that to be returned
            - return_all_metrics(boolean): whether to return all the metrics, default is True.
            - metrics(list): If return_all_metrics==False, return only the metrics in this list.

        @Output: csv formatted
        """
        with open('g.json') as f:
            models = [{ele['name']:ele['paras']} for ele in json.load(f)['methods']]
        query_instance = Query(collection)
        eval_instance = Evaluation(collection)
        for i in query_instance.get_queries_lengths():
            print '-'*20
            print i,':'
            qids = [q['num'] for q in query_instance.get_queries_of_length(i)]
            for para_key, para_value in models[0].items():
                model = para_key+','+para_value.iterkeys().next()+':'+str(para_value.itervalues().next()[0])
                print model
                print eval_instance.get_all_performance_of_some_queries(method = model, qids = qids, return_all_metrics = False)
        

PerformaceAnalysis().gen_performance_trending()