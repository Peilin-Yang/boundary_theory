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
    def __init__(self, corpus_path, result_file_path):
        self.corpus_path = os.path.abspath(corpus_path)
        self.result_file_path = os.path.abspath(result_file_path)
        if not os.path.exists(self.corpus_path) or not os.path.exists(self.result_file_path):
            print '[Evaluation Constructor]:Please provide valid corpus path and result file path'
            exit(1)

        self.judgment_file_path = os.path.join(self.corpus_path, 'judgment_file')
        if not os.path.exists(self.judgment_file_path):
            print """No judgment file found! 
                judgment file should be called "judgment_file" under 
                corpus path. You can create a symlink for it"""
            exit(1)


    def get_all_performance(self, return_all_metrics=True, metrics=['map']):
        """
        get all kinds of performance

        @Input:
            - return_all_metrics(boolean): whether to return all the metrics, default is True.
            - metrics(list): If return_all_metrics==False, return only the metrics in this list.

        @Return: a dict of performances 
        """


        cache_dir_path = os.path.join(self.corpus_path, 'all_performances')
        if not os.path.exists(cache_dir_path):
            os.makedirs(cache_dir_path)

        cache_file_path = os.path.join(cache_dir_path, os.path.basename(self.result_file_path))
        if not os.path.exists(cache_file_path):
            all_performances = {}
            process = Popen(['trec_eval', '-q', '-m', 'all_trec', self.judgment_file_path, self.result_file_path], stdout=PIPE)
            stdout, stderr = process.communicate()
            for line in stdout.split('\n'):
                line = line.strip()
                if line:
                    row = line.split()
                    evaluation_method = row[0]
                    qid = row[1]
                    try:
                        performace = ast.literal_eval(row[2])
                    except:
                        continue

                    if qid not in all_performances:
                        all_performances[qid] = {}
                    all_performances[qid][evaluation_method] = performace
            #print all_performances
            with open(cache_file_path, 'wb') as f:
                json.dump(all_performances, f, indent=2, sort_keys=True)
                f.flush()

        with open(cache_file_path) as f:
            j = json.load(f)
        if return_all_metrics:
            return j
        else:
            j = {qid:{m:j[qid][m] for m in metrics} for qid in j}
            return j



    def get_all_performance_of_some_queries(self, qids, return_all_metrics=True, metrics=['map']):
        """
        get all kinds of performance

        @Input:
            - qids (list) : a list contains the qid that to be returned
            - return_all_metrics(boolean): whether to return all the metrics, default is True.
            - metrics(list): If return_all_metrics==False, return only the metrics in this list.

        @Return: a dict of all performances of qids
        """

        all_performances = self.get_all_performance(return_all_metrics, metrics)
        return {k: all_performances.get(k, None) for k in qids}




if __name__ == '__main__':
    e = Evaluation('../../wt2g', '../../wt2g/results/tf1')
    print e.get_all_performance()
    raw_input()

