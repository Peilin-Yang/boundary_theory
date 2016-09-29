import sys, os
import json
from operator import itemgetter
import math
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
        self.output_root = '../../all_results/'
  
    def plot_para_trending(self, collection, query_part, metric, outfmt='png'):
        """
        Generate the performance trendings based on the parameters of 
        the model and the number of terms in the query
        @Input:
            @collection: path to the collections (evaluation results)
            @query_part: e.g. title, desc, title+desc
            @metric: e.g. map, p@20

        @Output: plots in files
        """
        # We assume that there is ONLY ONE parameter in the model!!
        collection_name = collection.split('/')[-1]
        json_output_fn = os.path.join(self.output_root, 
            'performance_analysis', collection_name+'_'+query_part+'_'+metric+'.json')
        with open('g.json') as f:
            models = [{ele['name']:ele['paras']} for ele in json.load(f)['methods']]
        query_instance = Query(collection)
        eval_instance = Evaluation(collection)
        query_nums = query_instance.get_queries_lengths(query_part)
        print collection_name, query_part, query_nums
        #plot related
        num_cols = 2
        num_rows = int(math.ceil(len(query_nums)*1.0/num_cols))
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, sharex=False, 
            sharey=False, figsize=(num_cols, num_rows))
        font = {'size' : 4}
        plt.rc('font', **font)
        row_idx = 0
        col_idx = 0
        for i in query_nums:
            qids = [q['num'] for q in query_instance.get_queries_of_length(i, query_part)]
            this_json_output = {'qLen': i, 'qids': qids}
            # we assume that the model parameters can be normalized to [0, 1]
            ax = axs[row_idx][col_idx]
            col_idx += 1
            if col_idx >= num_cols:
                row_idx += 1
                col_idx = 0
            markers = ['ro', 'bs', 'kv', 'gx'] 
            for model_idx, model in enumerate(models):
                for para_key, para_value in model.items():
                    avg_perform = []
                    orig_x = para_value.values()[0]
                    x = [para*1.0/max(para_value.values()[0]) for para in para_value.values()[0]]
                    for para in para_value.values()[0]:
                        method_str = para_key+','+para_value.iterkeys().next()+':'+str(para)
                        avg_perform.append( np.mean([v[metric] if v else 0.0 for v in eval_instance.get_all_performance_of_some_queries(method = method_str, qids = qids, return_all_metrics = False).values()]) )
                    ax.plot(x, avg_perform, markers[model_idx], ls='-', label=model.keys()[0])
                    zipped = zip(orig_x, x, avg_perform)
                    zipped.sort(key=itemgetter(2,1,0), reverse=True)
                    this_json_output['model'] = model.keys()[0]
                    this_json_output['para'] = zipped[0][0]
                    this_json_output['performance'] = zipped[0][2]
                    print model.keys()[0], zipped[0]
            ax.set_title('qLen=%d' % i)
        plt.legend()
        plt.savefig(os.path.join(self.output_root, 
            'performance_analysis', collection_name+'_'+query_part+'_'+metric+'.'+outfmt), 
            format=outfmt, bbox_inches='tight', dpi=400)    
