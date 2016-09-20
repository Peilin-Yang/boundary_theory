import sys,os
import ast
import json
import csv
from subprocess import Popen, PIPE


class Baselines(object):
    """
    Baselines related
    """
    def __init__(self, corpus_path):
        self.corpus_path = os.path.abspath(corpus_path)
        if not os.path.exists(self.corpus_path):
            print '[Baselines Constructor]:Please provide valid corpus path'
            exit(1)

    def gen_baseline_best_results(self):
        MAP = {}
        for fn in os.listdir(os.path.join(self.corpus_path, 'all_baseline_results')):
            method = fn.split('_')[0] 
            evaluation = Evaluation(self.corpus_path, os.path.join(self.corpus_path, 'all_baseline_results', fn)) \
                            .get_all_performance(return_all_metrics=True, metrics=['map'])
            for qid in evaluation:
                if qid not in MAP:
                    MAP[qid] = {}
                if method not in MAP[qid]:
                    MAP[qid][method] = {}
                MAP[qid][method][fn] = evaluation[qid]['map']
        MAP1 = {}
        for qid in MAP:
            MAP1[qid] = []
            for method in MAP[qid]:
                MAP1[qid].append(sorted(MAP[qid][method].items(), key=itemgetter(1), reverse=True)[0])

        with open(os.path.join(self.corpus_path, 'all_baseline_results.csv'), 'wb') as output:
            for qid in sorted(MAP1.keys()):
                output.write('%s' % qid)
                for ele in MAP1[qid]:
                    output.write(',%s,%s' % (ele[0], ele[1]))
                output.write('\n')



    def get_baseline_best_results(self):
        """
        @Return: a dict of best results for each query
        """
        fn = os.path.join(self.corpus_path, 'all_baseline_results.csv')
        if not os.path.exists( fn ):
            self.gen_baseline_best_results()
        r = {}
        with open(fn) as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                qid = row[0]
                pivoted_best = ast.literal_eval(row[2])
                lm_best = ast.literal_eval(row[4])
                okapi_best = ast.literal_eval(row[6])
                r[qid] = max([pivoted_best, lm_best, okapi_best])
        return r





if __name__ == '__main__':
    e = Evaluation('../../wt2g', '../../wt2g/results/tf1')
    print e.get_all_performance()
    raw_input()

