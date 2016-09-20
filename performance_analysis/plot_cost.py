import sys, os
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
        query_instance = Query(collection)
        for i in query_instance.get_queries_lengths():
            print '-'*20
            print i,':'
            print query_instance.get_queries_of_length(i)
        

PerformaceAnalysis().gen_performance_trending()