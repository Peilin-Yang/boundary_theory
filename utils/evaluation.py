import sys,os
import ast
import json
from subprocess import Popen, PIPE


class Evaluation():
    """
    Get the evaluation of a corpus for a result.
    When constructing, pass the path of the corpus and the path of the result file. 
    For example, "../wt2g/" "../wt2g/results/idf1"
    """
    def __init__(self, corpus_path):
        self.corpus_path = os.path.abspath(corpus_path)
        if not os.path.exists(self.corpus_path):
            print '[Evaluation Constructor]:Please provide valid corpus path'
            exit(1)

        self.judgment_file_path = os.path.join(self.corpus_path, 'judgement_file')
        if not os.path.exists(self.judgment_file_path):
            print """No judgment file found! 
                judgment file should be called "judgment_file" under 
                corpus path. You can create a symlink for it"""
            exit(1)
        self.evaluations_path = os.path.join(self.corpus_path, 'evals')
        if not os.path.exists(self.evaluations_path):
            os.makedirs(self.evaluations_path)


    def get_all_performance(self, method='lm', qf_parts='title', return_all_metrics=True, metrics=['map']):
        """
        get all kinds of performance

        @Input:
            - method: which method to return. coubld be with parameters
            - qf_parts: results using which part of the query, e.g. title, description, narrative and so on.
            - return_all_metrics(boolean): whether to return all the metrics, default is True.
            - metrics(list): If return_all_metrics==False, return only the metrics in this list.

        @Return: a dict of performances 
        """
        fn = os.path.join(self.evaluations_path, qf_parts+'-method:'+method)

        with open(fn) as f:
            j = json.load(f)
        if return_all_metrics:
            return j
        else:
            j = {qid:{m:j[qid][m] for m in metrics} for qid in j}
            return j



    def get_all_performance_of_some_queries(self, method='lm', qf_parts='title', qids=[], return_all_metrics=True, metrics=['map']):
        """
        get all kinds of performance

        @Input:
            - qids (list) : a list contains the qid that to be returned
            - return_all_metrics(boolean): whether to return all the metrics, default is True.
            - metrics(list): If return_all_metrics==False, return only the metrics in this list.

        @Return: a dict of all performances of qids
        """

        all_performances = self.get_all_performance(method, qf_parts, return_all_metrics, metrics)
        return {k: all_performances.get(k, None) for k in qids}




if __name__ == '__main__':
    e = Evaluation('../../wt2g', '../../wt2g/results/tf1')
    print e.get_all_performance()
    raw_input()

