import sys,os
import math
import argparse
import json
import ast
import subprocess
import time
from subprocess import Popen, PIPE
from datetime import datetime
from operator import itemgetter
import multiprocessing
import re

import inspect
from inspect import currentframe, getframeinfo

import numpy as np
from scipy.stats import entropy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


#from sklearn import datasets, linear_model

from utils.query import Query
from utils.results_file import ResultsFile
from utils.judgment import Judgment
from utils.evaluation import Evaluation
from utils.utils import Utils
from utils.collection_stats import CollectionStats
from utils.baselines import Baselines
from utils import indri
from utils.MPI import MPI


def IndriRunQuery(query_file_path, output_path, method=None):
    """
    This function should be outside the class so that it can be executed by 
    children process.
    """
    frameinfo = getframeinfo(currentframe())
    print frameinfo.filename+':'+str(frameinfo.lineno),
    print query_file_path, method, output_path
    with open(output_path, 'wb') as f:
        if method:
            subprocess.Popen(['IndriRunQuery', query_file_path, method], bufsize=-1, stdout=f)
        else:
            subprocess.Popen(['IndriRunQuery', query_file_path], bufsize=-1, stdout=f)
        f.flush()
        os.fsync(f.fileno())
        time.sleep(3)


def process_json(c, r):
    json_results = {}
    c_tag = c[3:]
    #print c_tag
    cs = CollectionStats(c)
    doc_cnt = cs.get_doc_counts()
    single_queries = Query(c).get_queries_of_length(1)
    qids = [ele['num'] for ele in single_queries]
    #print qids
    judgment = Judgment(c).get_relevant_docs_of_some_queries(qids, 1, 'dict')
    r_tag = r
    #print r_tag
    results = ResultsFile(os.path.join(c, r)).get_results_of_some_queries(qids)
    #print qids, results.keys()
    for qid, qid_results in results.items():
        this_key = c+','+qid+','+r_tag
        json_results[this_key] = []
        non_rel_cnt = 0
        #print qid
        qid_doc_stats = cs.get_qid_doc_statistics(qid)
        maxTF = cs.get_term_maxTF(cs.get_idf(qid).split('-')[0])
        for idx, ele in enumerate(qid_results):
            docid = ele[0]
            score = ele[1]
            if docid in judgment[qid]:
                json_results[this_key].append(\
                    (docid, score, qid_doc_stats[docid]['TOTAL_TF'], \
                    maxTF, non_rel_cnt, non_rel_cnt*1./doc_cnt))
            else:
                non_rel_cnt += 1
    #print json_results
    return json_results

def get_external_docno(collection_path, docid):
    return CollectionStats(collection_path).get_external_docid(docid)


def callwrapper(func, args):
    return func(*args)

def pool_call(args):
    return callwrapper(*args)


class SingleQueryAnalysis():
    def __init__(self):
        self.all_results_root = '../all_results'
        self.batch_root = '../batch/'


    def plot_single(self, collection='../wt2g', 
        method='pivotedwithoutidf_0.3', qid='417', outputformat='png'):
        """
        plot a single query, e.g. pivoted-wt2g-417.
        """
        output_root = '../output/single_term_query_analysis/'

        for fn in os.listdir(output_root):
            if re.search(r'json$', fn):
                json_results_file_path = os.path.join(output_root, fn)
                break

        with open(json_results_file_path) as f:
            json_results = json.load(f)
        
        level1_key = collection+','+qid
        level2_key = os.path.join('results', method)

        print level1_key, level2_key

        if level1_key not in json_results:
            print 'Collection or Qid not exists in the result file:'+json_results_file_path 
            exit()
        if level2_key not in json_results[level1_key]:
            print 'Method not exists in the result file:'+json_results_file_path 
            exit()

        rel_docs_stats = json_results[level1_key][level2_key]

        x_vals = [ele[2]*1./ele[3] for ele in rel_docs_stats if ele[2]!=0 and ele[5]!=0]
        y_vals = [ele[5] for ele in rel_docs_stats if ele[2]!=0 and ele[5]!=0]

        clf = linear_model.LinearRegression()
        #print x_vals
        #print y_vals
        x_linear = [[math.log10(x)] for x in x_vals]
        y_linear = [[math.log10(y)] for y in y_vals]

        clf.fit(x_linear, y_linear)
        #print clf.predict(x_vals, y_vals)
        print clf.coef_, clf.intercept_
        #raw_input()
        x_linear_plot = np.arange(min(x_vals), 1.0, 0.001)
        #y_linear_plot = x_linear_plot*clf.coef_[0][0]+clf.intercept_[0]
        y_linear_plot = np.power([10 for ele in x_linear_plot], np.log10(x_linear_plot)*clf.coef_[0][0]+clf.intercept_[0])


        plt.plot(x_vals, y_vals, 'r.')
        plt.plot(x_linear_plot, y_linear_plot, 'r-')
        plt.xscale('log')
        plt.yscale('log')
        output_root = '../output/single_term_query_analysis/'
        collection_tag = collection[3:]
        ofn = os.path.join(output_root, collection_tag+'_'+method+'_'+qid+'.'+outputformat)
        plt.savefig(ofn, format=outputformat, bbox_inches='tight', dpi=400)



    def plot_cost(self, json_results, outputformat='png'):
        """
        Plot based on the json_results
        """

        all_collections = []
        all_methods_name = []
        k, v = json_results.iteritems().next()
        for m in v:
            method = m.split('/')[-1].split('_')[0]
            if method not in all_methods_name:
                all_methods_name.append(method)

        collection_separate_results = {}
        collection_stats = {}
        for k in sorted(json_results.keys()):
            collection_path, qid = k.split(',')
            if collection_path not in collection_stats:
                collection_stats[collection_path] = CollectionStats(collection_path).get_richStats()
                allTerms = collection_stats[collection_path]['allTerms'].values()
                allDocLens = collection_stats[collection_path]['allDocs'].values()
                del(collection_stats[collection_path]['allDocs'])
                del(collection_stats[collection_path]['allTerms'])
                del(collection_stats[collection_path]['total terms'])
                del(collection_stats[collection_path]['average doc length'])
                del(collection_stats[collection_path]['unique terms'])
                collection_stats[collection_path]['avg_doc_len'] = np.average(allDocLens)
                collection_stats[collection_path]['max_doc_len'] = max(allDocLens)
                collection_stats[collection_path]['var_doc_len'] = np.var(allDocLens)
                collection_stats[collection_path]['avg_tf'] = np.average(allTerms)
                collection_stats[collection_path]['max_tf'] = max(allTerms)
                collection_stats[collection_path]['var_tf'] = np.var(allTerms)
            if collection_path not in collection_separate_results:
                collection_separate_results[collection_path] = {}
            collection_separate_results[collection_path][qid] = json_results[k]

        #print collection_separate_results['../trec7']['395']
        #raw_input()
        #print all_methods_name
        #print collection_stats
        #exit()

        for m in all_methods_name:
            for collection_path in sorted(collection_separate_results):
                collection_tag = collection_path[3:]
                qid_results = collection_separate_results[collection_path]

                num_cols = 4
                num_rows = int(math.ceil((len(qid_results)*2.0+2)/num_cols)) # additional one for lengend

                fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, sharex=True, sharey=False, figsize=(3.*num_cols, 3.*num_rows))
                row_idx = 0
                col_idx = 0
                font = {'size' : 8}
                plt.rc('font', **font)

                formats = ['r.', 'b1', 'g+']
                linecolors = ['r', 'b', 'g']

                all_idf = []
                for qid in sorted(qid_results):
                    cs = CollectionStats(collection_path)
                    idf = cs.get_idf(qid)
                    idf_value = float(idf.split('-')[1])
                    query_term = idf.split('-')[0]
                    termCounts = [int(ele[1]) for ele in cs.get_term_counts(query_term)]
                    all_idf.append((qid, idf_value, idf, termCounts))
                    query_term = idf.split('-')[0]
                all_idf.sort(key=itemgetter(1,0))
                for ele in all_idf:
                    qid = ele[0]
                    idf = ele[2]
                    termCounts = ele[3]
                    #print termCounts
                    if num_rows == 1:
                        ax = axs[col_idx]
                    else:
                        ax = axs[row_idx, col_idx]

                    
                    LinearRegressionText = ''

                    format_idx = 0
                    all_methods = []
                    performance_list = []
                    for method_path in sorted(qid_results[qid]):
                        rel_docs_stats = qid_results[qid][method_path]
                        method = method_path.split('/')[-1]
                        if not re.match(r'^%s'%m, method):
                            continue
                        #print collection_path, os.path.join(collection_path, method_path)
                        evaluation = Evaluation(collection_path, os.path.join(collection_path, method_path))\
                            .get_all_performance_of_some_queries([qid])

                        if method.find('withoutidf') >= 0:
                            if len(method.split('_')) > 1:
                                method = method.split('_')[0][:method.find('withoutidf')]+'_'+method.split('_')[1]
                        performance_list.append(method+':'+str(evaluation[qid]['map']))
                        all_methods.append(method)
                        x_vals = [ele[2]*1./ele[3] for ele in rel_docs_stats if ele[2]!=0 and ele[5]!=0]
                        y_vals = [ele[5] for ele in rel_docs_stats if ele[2]!=0 and ele[5]!=0]

                        # Linear Regression
                        clf = linear_model.LinearRegression()
                        #print x_vals
                        #print y_vals
                        x_linear = []
                        y_linear = []
                        
                        for x in x_vals:
                            x_linear.append([math.log10(x)])
                        for y in y_vals:
                            y_linear.append([math.log10(y)])

                        """
                        for x in x_vals:
                            x_linear.append([x])
                        for y in y_vals:
                            y_linear.append([y])
                        """

                        clf.fit(x_linear, y_linear)
                        print 'Variance score: %.2f' % clf.score(x_linear, y_linear)
                        print 'get_params: %s' % repr(clf.get_params())
                        print 'decision_function: %s' % repr(clf.decision_function(x_linear))
                        #print clf.predict(x_vals, y_vals)
                        #print clf.coef_, clf.intercept_
                        #raw_input()
                        x_linear_plot = np.arange(min(x_vals), 1.0, 0.001)
                        #y_linear_plot = x_linear_plot*clf.coef_[0][0]+clf.intercept_[0]
                        y_linear_plot = np.power([10 for ele in x_linear_plot], np.log10(x_linear_plot)*clf.coef_[0][0]+clf.intercept_[0])
                        #y_linear_plot = np.power([0.001 for ele in x_linear_plot], x_linear_plot)
                        #print x_linear_plot
                        #print y_linear_plot
                        LinearRegressionText += 'coef_('+linecolors[format_idx]+'):'+str(round(clf.coef_[0][0], 6))+'\n'
                        LinearRegressionText += 'intercept_('+linecolors[format_idx]+'):'+str(round(clf.intercept_[0], 6))+'\n'

                        #print x_linear_plot
                        #print y_linear_plot
                        # Linear Regression

                        ax.plot(x_vals, y_vals, formats[format_idx])
                        ax.plot(x_linear_plot,  y_linear_plot, c=linecolors[format_idx], ls='-', lw='1.5')
                        format_idx += 1

                    ax.set_xscale('log')
                    ax.set_yscale('log')
                    ax.set_title(collection_tag+' '+qid)
                    col_idx += 1
                    if col_idx == num_cols:
                        row_idx += 1
                        col_idx = 0

                    # plot text 
                    if num_rows == 1:
                        ax = axs[col_idx]
                    else:
                        ax = axs[row_idx, col_idx]

                    _text = 'idf:'+str(idf)+'(N/df)'+'\n'
                    _text += '\n'
                    _text += 'Performace'+'\n'
                    _text += '\n'.join(performance_list)+'\n'
                    _text += '\n'
                    _text += 'Statistics'+'\n'
                    _text += 'average TF of query terms:'+str(round(np.average(termCounts), 2)) + '\n'
                    _text += 'max TF of query terms:'+str(max(termCounts)) + '\n'
                    _text += 'variance TF of query terms:'+str(round(np.var(termCounts), 2)) + '\n'
                    _text += '\n'
                    _text += 'LinearRegression'+'\n'
                    _text += LinearRegressionText
                    _text = _text.strip()
                    ax.text(0.05, 0.5, _text, bbox=dict(facecolor='none', alpha=0.8),\
                     horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
                    ax.axis('off')

                    col_idx += 1
                    if col_idx == num_cols:
                        row_idx += 1
                        col_idx = 0

                if num_rows == 1:
                    ax = axs[col_idx]
                else:
                    ax = axs[row_idx, col_idx]
                for i, method in enumerate(all_methods):
                    ax.plot([0], [0], formats[i], label=method)
                    t = '# of docs:'+str(collection_stats[collection_path]['documents'])+'\n'
                    t += 'avg doc len:'+str(round(collection_stats[collection_path]['avg_doc_len'], 2))+'\n'
                    t += 'max doc len:'+str(collection_stats[collection_path]['max_doc_len'])+'\n'
                    t += 'variance doc len:'+str(round(collection_stats[collection_path]['var_doc_len'], 2))+'\n'
                    t += 'avg TF:'+str(round(collection_stats[collection_path]['avg_tf'], 2))+'\n'
                    t += 'max TF:'+str(collection_stats[collection_path]['max_tf'])+'\n'
                    t += 'variance TF:'+str(round(collection_stats[collection_path]['var_tf'], 2))
                    ax.text(0.05, 0.25, t,\
                     bbox=dict(facecolor='none', alpha=0.8),\
                     horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
                    ax.axis('off')
                ax.legend()

                col_idx += 1
                if col_idx == num_cols:
                    row_idx += 1
                    col_idx = 0


                # some explainations
                if num_rows == 1:
                    ax = axs[col_idx]
                else:
                    ax = axs[row_idx, col_idx]

                explaination = 'avg TF = \n  SUM(term counts in collection)\n'+\
                    '------------------------------------------------\n  (# of unique terms in collection)\n' 
                explaination += '\n'  
                explaination += 'avg TF of query term = \n  SUM(term counts in documents)\n'+\
                    '------------------------------------------------\n  (# of documents contains query term)\n'
                ax.text(0., 0.5, explaination, bbox=dict(facecolor='none', alpha=0.8),\
                     horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
                ax.axis('off')

                #ax.set_xticks([])
                #ax.set_yticks([])
                #ax.set_xlim([0., 1])
                #ax.set_ylim([0.000001, 0.1])
                #ax.set_ylabel('# of non-relevant docs/# of total docs in collection')

                fig.text(0.08, 0.5, '# of non-relevant docs/# of total docs in collection', ha='center', va='center', rotation='vertical')
                fig.text(0.5, 0.0, 'TF/MAX_TF', ha='center', va='center')
                #plt.savefig(os.path.join(output_root, 'detailed_qids.png'), format='png', dpi=400)

                output_root = '../output/single_term_query_analysis/'
                ofn = os.path.join(output_root, '_single_term_plots_'+collection_tag+'_'+m+'_diffX_sameY_X_normMaxTFLog.'+outputformat)
                plt.savefig(ofn, format=outputformat, bbox_inches='tight', dpi=400)


    def plot_cost_of_rel_docs(self):
        """
        Plot the cost of retrieving relevant documents.
        The cost of retrieving a relevant document is the number of 
        non-relevant documents before it.
        """

        #print datetime.now()
        output_root = '../output/single_term_query_analysis/'
        if not os.path.exists(output_root):
            os.makedirs(output_root)

        corpus_list = ['../wt2g', '../trec8', '../trec7']

        """
        results_list = [
            'all_baseline_results/okapi_0.05', 
            'all_baseline_results/pivoted_0.05',
            'results/tf1',
            'results/ln1'
        ]
        """
        results_list = [
            'results/okapiwithoutidf_0.05', 
            'results/okapiwithoutidf_0.5', 
            'results/okapiwithoutidf_0.7', 
            'results/pivotedwithoutidf_0.05',
            'results/pivotedwithoutidf_0.2',
            'results/pivotedwithoutidf_0.3',
            'results/tf1',
            'results/ln1'
        ]
        tag = '-'.join([ele.split('/')[-1] for ele in results_list])
        ofn = os.path.join(output_root, 'cost_of_rel_docs_'+tag+'.json')
        if not os.path.exists(ofn):
            json_results = []
            pool = multiprocessing.Pool(16)
            paras = []
            for c in corpus_list:
                for r in results_list:
                    paras.append((process_json, (c, r)))
            #print tt

            r = pool.map_async(pool_call, paras, callback=json_results.extend)
            r.wait()

            output = {}
            for ele in json_results:
                for k, v in ele.items():
                    output_key = ','.join(k.split(',')[:-1])
                    if output_key not in output:
                        output[output_key] = {}
                    output[output_key][k.split(',')[-1]] = v

            with open(ofn, 'wb') as f:
                json.dump(output, f, indent=4)
                        
            """
            # get all queryies with single term
            json_results = {}
            for c in corpus_list:
                c_tag = c[3:]
                #print c_tag
                doc_cnt = CollectionStats(c).get_doc_counts()
                single_queries = Query(c).get_queries_of_length(1)
                qids = [ele['num'] for ele in single_queries]
                #print qids
                judgment = Judgment(c).get_relevant_docs_of_some_queries(qids, 1, 'dict')
                for r in results_list:
                    #r_tag = r.split('/')[1]
                    r_tag = r
                    print r_tag
                    results = ResultsFile(os.path.join(c, r)).get_results_of_some_queries(qids)
                    #print qids, results.keys()
                    for qid, qid_results in results.items():
                        this_key = c+','+qid
                        if this_key not in json_results:
                            json_results[this_key] = {}
                        if r_tag not in json_results[this_key]:
                            json_results[this_key][r_tag] = []
                        non_rel_cnt = 0
                        print qid
                        qid_doc_stats = CollectionStats(c).get_qid_doc_statistics(qid)
                        for idx, ele in enumerate(qid_results):
                            docid = ele[0]
                            score = ele[1]
                            if docid in judgment[qid]:
                                json_results[this_key][r_tag].append(\
                                    (docid, score, qid_doc_stats[docid]['TOTAL_TF'], \
                                    non_rel_cnt, non_rel_cnt*1./doc_cnt))
                            else:
                                non_rel_cnt += 1
                                
            with open(ofn, 'wb') as f:
                json.dump(json_results, f, indent=4)
            """
        #print datetime.now()
        with open(ofn) as f:
            json_results = json.load(f)

        self.plot_cost(json_results)




    def batch_run_okapi_pivoted_without_idf(self):
        corpus_list = ['../wt2g', '../trec8', '../trec7']
        children_process = []
        for c in corpus_list:
            for i in np.arange(0, 1.01, 0.05):
                children_process.append([os.path.join(os.path.abspath(c), 'standard_queries'), \
                    os.path.join(c, 'results', 'pivotedwithoutidf_'+str(i)), '-rule=method:pivotedwithoutidf,s:'++str(i)])
                children_process.append([os.path.join(os.path.abspath(c), 'standard_queries'), \
                    os.path.join(c, 'results', 'okapiwithoutidf_'+str(i)), '-rule=method:okapiwithoutidf,b:'+str(i)])
            
        print children_process
        #exit()
        Utils().run_multiprocess_program(IndriRunQuery, children_process)



    def plot_single_tfc_constraints_draw_pdf(self, ax, xaxis, yaxis, 
            title, legend, legend_outside=False, xlog=True, ylog=False):
        # 1. probability distribution 
        ax.plot(xaxis, yaxis, 'ro', ms=3.5, label=legend)
        ax.vlines(xaxis, [0], yaxis)
        if xlog:
            ax.set_xscale('log')
        if ylog:
            ax.set_yscale('log')
        ax.set_xlim(0, ax.get_xlim()[1])
        ax.set_ylim(0, ax.get_ylim()[1] if ax.get_ylim()[1]!=1 else 1.02)
        ax.set_title(title)
        if legend_outside:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            ax.legend()

    def plot_single_tfc_constraints_draw_hist(self, ax, yaxis, nbins, _norm, title, legend):
        #2. hist gram
        yaxis.sort()
        n, bins, patches = ax.hist(yaxis, nbins, normed=_norm, facecolor='#F08080', alpha=0.5, label=legend)
        ax.set_title(title)
        ax.legend()


    def plot_single_tfc_constraints_tf_rel(self, collection_path, smoothing=True):
        collection_name = collection_path.split('/')[-1]
        cs = CollectionStats(collection_path)
        output_root = 'single_query_figures'
        single_queries = Query(collection_path).get_queries_of_length(1)
        queries = {ele['num']:ele['title'] for ele in single_queries}
        #print qids
        rel_docs = Judgment(collection_path).get_relevant_docs_of_some_queries(queries.keys(), 1, 'dict')
        #print rel_docs
        #raw_input()
        collection_level_tfs = []
        collection_level_x_dict = {}
        collection_level_maxTF = 0
        fig, axs = plt.subplots(nrows=len(rel_docs), ncols=2, sharex=False, sharey=False, figsize=(6*2, 3.*len(rel_docs)))
        font = {'size' : 8}
        plt.rc('font', **font)
        row_idx = 0
        for i, qid in enumerate(rel_docs):
            query_term = queries[qid]
            maxTF = cs.get_term_maxTF(query_term)
            if maxTF > collection_level_maxTF:
                collection_level_maxTF = maxTF
            #print maxTF
            idf = cs.get_term_IDF1(query_term)
            tfs = [int(tf) for tf in cs.get_term_docs_tf_of_term_with_qid(qid, query_term, rel_docs[qid].keys())]
            rel_docs_len = len( rel_docs[qid].keys() )
            #print tfs, rel_docs_len
            doc_with_zero_tf_len = len( rel_docs[qid].keys() ) - len(tfs)
            tfs.extend([0]*doc_with_zero_tf_len)
            collection_level_tfs.extend(tfs)
            #print len( rel_docs[qid].keys() )
            #print len(tfs)
            x_dict = {}
            for tf in tfs:
                if tf not in x_dict:
                    x_dict[tf] = 0
                x_dict[tf] += 1
                if tf not in collection_level_x_dict:
                    collection_level_x_dict[tf] = 0
                collection_level_x_dict[tf] += 1
            x_dict[0] = len( rel_docs[qid].keys() ) - len(tfs)
            yaxis_hist = tfs
            yaxis_hist.sort()
            #print len(yaxis_hist)
            #print yaxis_hist
            for tf in range(0, maxTF+1):
                if tf not in x_dict:
                    x_dict[tf] = 0
                if smoothing:
                    x_dict[tf] += .1

                if tf not in collection_level_x_dict:
                    collection_level_x_dict[tf] = 0
                if smoothing:
                    collection_level_x_dict[tf] += .1

            xaxis = x_dict.keys()
            xaxis.sort()
            yaxis_pdf = [x_dict[x]/rel_docs_len for x in xaxis]

            self.plot_single_tfc_constraints_draw_pdf(axs[row_idx][0], xaxis, 
                yaxis_pdf, qid+'-'+query_term, 
                "maxTF=%d\n|rel_docs|=%d\nidf=%f" % (maxTF, rel_docs_len, idf), 
                ylog=False)
            self.plot_single_tfc_constraints_draw_hist(axs[row_idx][1], yaxis_hist, 
                math.ceil(maxTF/10.), False, qid+'-'+query_term, 
                '#bins(maxTF/10.0)=%d' % (math.ceil(maxTF/10.)))
            row_idx += 1

        fig.text(0.5, 0.07, 'Term Frequency', ha='center', va='center', fontsize=12)
        fig.text(0.06, 0.5, 'P( c(t,D)=x | D is a relevant document)=tf/|rel_docs|', ha='center', va='center', rotation='vertical', fontsize=12)
        fig.text(0.5, 0.5, 'Histgram', ha='center', va='center', rotation='vertical', fontsize=12)
        fig.text(0.6, 0.04, 'Histgram:rel docs are binned by their TFs. The length of the bin is set to 10. Y axis shows the number of rel docs in each bin.', ha='center', va='center', fontsize=10)

        plt.savefig(os.path.join(self.all_results_root, output_root, collection_name+'-tf_rel.png'), 
            format='png', bbox_inches='tight', dpi=400)

        #collection level
        collection_level_xaxis = collection_level_x_dict.keys()
        collection_level_xaxis.sort()
        collection_level_yaxis_pdf = [collection_level_x_dict[x]/len(collection_level_tfs) for x in collection_level_xaxis]

        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False, figsize=(6, 2.*2))
        font = {'size' : 8}
        plt.rc('font', **font)
        self.plot_single_tfc_constraints_draw_pdf(axs[0], collection_level_xaxis, 
            collection_level_yaxis_pdf, collection_name, "", ylog=False)
        self.plot_single_tfc_constraints_draw_hist(axs[1], collection_level_tfs, 
            math.ceil(collection_level_maxTF/10.), False, "", 
            '#bins(collection_level_maxTF/10.0)=%d' % (math.ceil(collection_level_maxTF/10.)))
        #fig.text(0.5, 0.07, 'Term Frequency', ha='center', va='center', fontsize=12)
        #fig.text(0.06, 0.5, 'P( c(t,D)=x | D is a relevant document)=tf/|rel_docs|', ha='center', va='center', rotation='vertical', fontsize=12)
        #fig.text(0.5, 0.5, 'Histgram', ha='center', va='center', rotation='vertical', fontsize=12)
        #fig.text(0.6, 0.04, 'Histgram:rel docs are binned by their TFs. The length of the bin is set to 10. Y axis shows the number of rel docs in each bin.', ha='center', va='center', fontsize=10)

        plt.savefig(os.path.join(self.all_results_root, output_root, collection_name+'-all-tf_rel.png'), 
            format='png', bbox_inches='tight', dpi=400)



    def plot_single_tfc_constraints_rel_tf(self, collection_path, smoothing=False):
        collection_name = collection_path.split('/')[-1]
        cs = CollectionStats(collection_path)
        output_root = 'single_query_figures'
        single_queries = Query(collection_path).get_queries_of_length(1)
        queries = {ele['num']:ele['title'] for ele in single_queries}
        #print qids
        rel_docs = Judgment(collection_path).get_relevant_docs_of_some_queries(queries.keys(), 1, 'dict')
        #print rel_docs

        collection_level_x_dict = {}
        collection_level_maxTF = 0

        # we draw two columns: 1. probability distribution 2. histgram
        fig, axs = plt.subplots(nrows=len(rel_docs), ncols=1, sharex=False, sharey=False, figsize=(6, 3.*len(rel_docs)))
        font = {'size' : 8}
        plt.rc('font', **font)
        row_idx = 0
        for qid in rel_docs:
            query_term = queries[qid]
            maxTF = cs.get_term_maxTF(query_term)
            if maxTF > collection_level_maxTF:
                collection_level_maxTF = maxTF
            #query_term_ivlist = cs.get_term_counts_dict(query_term)
            # detailed_stats_fn = os.path.join(collection_path, 'docs_statistics_json', qid)
            # with open(detailed_stats_fn) as f:
            #     detailed_stats_json = json.load(f)
            # tf_dict = {}
            # tf_dict = {k:[v['TOTAL_TF'], k in rel_docs[qid]] for k,v in detailed_stats_json.items()}
            # x_dict = {}
            # for docid, values in tf_dict.items():
            x_dict = {}
            qid_docs_len = 0
            for row in cs.get_qid_details(qid):
                qid_docs_len += 1
                tf = int(row['total_tf'])
                rel = (int(row['rel_score'])>=1)
                if tf not in x_dict:
                    x_dict[tf] = [0, 0] # [rel_docs, total_docs]
                if rel:
                    x_dict[tf][0] += 1
                x_dict[tf][1] += 1

                if tf not in collection_level_x_dict:
                    collection_level_x_dict[tf] = [0, 0] # [rel_docs, total_docs]
                if rel:
                    collection_level_x_dict[tf][0] += 1
                collection_level_x_dict[tf][1] += 1

            xaxis = x_dict.keys()
            xaxis.sort()
            yaxis = [x_dict[x][0]*1./x_dict[x][1] for x in xaxis]
            #print xaxis
            #print yaxis
            xaxis_splits_10 = [[x for x in xaxis if x <= i+10 and x > i] for i in range(0, maxTF+1, 10)]
            #print xaxis_splits_10
            yaxis_splits_10 = [[x_dict[x][0]*1./x_dict[x][1] for x in xaxis if x <= i+10 and x > i] for i in range(0, maxTF+1, 10)]
            #print yaxis_splits_10
            entropy_splits_10 = [entropy(ele, base=2) for ele in yaxis_splits_10]
            query_stat = cs.get_term_stats(query_term)
            dist_entropy = entropy(yaxis, base=2)
            self.plot_single_tfc_constraints_draw_pdf(axs[row_idx], xaxis, 
                yaxis, qid+'-'+query_term, 
                'term_maxTF=%d\nterm_minTF=%d\nterm_avgTF=%.2f\nterm_varTF=%.2f\ndf=%d\ndist_entropy=%.2f\nsplit_entropy_10=%s'
                % (maxTF, query_stat['minTF'], query_stat['avgTF'], query_stat['varTF'],
                  query_stat['df'], dist_entropy, str(entropy_splits_10)), 
                True,
                ylog=False)
            row_idx += 1

        collection_vocablulary_stat = cs.get_vocabulary_stats()
        collection_vocablulary_stat_str = ''
        idx = 1
        for k,v in collection_vocablulary_stat.items():
            collection_vocablulary_stat_str += k+'='+'%.2f'%v+' '
            if idx == 3:
                collection_vocablulary_stat_str += '\n'
                idx = 1
            idx += 1
        fig.text(0.5, 0, collection_vocablulary_stat_str, ha='center', va='center', fontsize=12)
        plt.savefig(os.path.join(self.all_results_root, output_root, collection_name+'-rel_tf.png'), 
            format='png', bbox_inches='tight', dpi=400)


        fig, axs = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(6, 3.*1))
        font = {'size' : 8}
        plt.rc('font', **font)
        xaxis = collection_level_x_dict.keys()
        xaxis.sort()
        yaxis = [collection_level_x_dict[x][0]*1./collection_level_x_dict[x][1] for x in xaxis]
        self.plot_single_tfc_constraints_draw_pdf(axs, xaxis, 
            yaxis, collection_name, 
            "collection_level_maxTF=%d" % (collection_level_maxTF), True,
            ylog=False)
        plt.savefig(os.path.join(self.all_results_root, output_root, collection_name+'-all-rel_tf.png'), 
            format='png', bbox_inches='tight', dpi=400)


    def plot_single_tfc_constraints(self, corpus_path):
        self.plot_single_tfc_constraints_tf_rel(corpus_path)
        self.plot_single_tfc_constraints_rel_tf(corpus_path)


    def plot_tfc_constraints(self, collections_path=[], smoothing=True):
        """
        * Start with the relevant document distribution according to one 
        term statistic and see how it affects the performance. 

        Take TF as an example and we start from the simplest case when 
        |Q|=1.  We could leverage an existing collection 
        and estimate P( c(t,D)=x | D is a relevant document), 
        where x = 0,1,2,...maxTF(t).

        Note that this step is function-independent.  We are looking at
        a general constraint for TF (i.e., TFC1), and to see how well 
        real collections would satisfy this constraint and then formalize
        the impact of TF on performance based on the rel doc distribution. 
        """
        for c in collections_path:
            self.plot_single_tfc_constraints(c)


    #####################################
    def pre_screen_ax(self, performances_comparisons):
        """
        pre-screen the queries: 
        1. first sort their best MAP performances.
        2. then group 4 to one figure.
        """
        r = []

        xaxis = np.arange(1, 101)
        for qid in performances_comparisons:
            yaxis = [performances_comparisons[qid][x] for x in xaxis]
            best_map = max(yaxis)
            best_map_x = yaxis.index(best_map)+1 # best map's x coordinate
            r.append([qid, yaxis, best_map, best_map_x])

        r.sort(key=itemgetter(2, 0))
        return r



    def plot_compare_tf3_performances(self, performances_comparisons, collection_path):
        baseline_best_results = Baselines(collection_path).get_baseline_best_results()
        #print baseline_best_results
        collection_name = collection_path.split('/')[-1]
        screened_queries = self.pre_screen_ax(performances_comparisons)
        rows_cnt = int( math.ceil( len(performances_comparisons)/4. ) )
        fig, axs = plt.subplots(nrows=rows_cnt, ncols=1, sharex=False, sharey=False, figsize=(6, 3.*rows_cnt))
        font = {'size' : 8}
        plt.rc('font', **font)
        xaxis = np.arange(1, 101)
        if rows_cnt > 1:
            ax = axs[0]
        else:
            ax = axs
        idx = 0
        ax_idx = 0
        cs = CollectionStats(collection_path)
        single_queries = Query(collection_path).get_queries_of_length(1)
        queries = {ele['num']:ele['title'] for ele in single_queries}

        for ele in screened_queries:
            qid = ele[0]
            best_map = ele[2]
            best_map_x = ele[3]
            maxTF = cs.get_term_maxTF(queries[qid])
            xaxis = np.arange(1, min(maxTF+1, 101) )
            yaxis = ele[1][:len(xaxis)]
            #print len(xaxis), len(yaxis)
            line, = ax.plot(xaxis, yaxis, label=qid)
            line_color = line.get_color()
            ax.vlines(best_map_x, 0, best_map, linestyles='dotted', linewidth=.5, colors=line_color)

            ax.hlines(baseline_best_results[qid], 0, maxTF+1, linestyles='-', linewidth=.5, colors=line_color)
            idx += 1
            if idx >= 4:
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                ax_idx += 1
                ax = axs[ax_idx]
                idx = 0
            if ax_idx == 0:
                ax.set_title("TF analysis for single term queries (%s)" % collection_name)
            ax.set_ylabel("Mean Average Precision")
            
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_xlabel("TF fade points: if TF<=fade point then Score=TF, if TF>fade point then Score=fade_point+log(TF)")
        plt.savefig( os.path.join(self.all_results_root, 'tf3_compare', collection_name+'-compare_tf3_performances.png'), 
            format='png', bbox_inches='tight', dpi=400)

    def compare_tf3_with_baselines(self, performances_comparisons, collection_path):
        output_path = 'tf3_compare'
        baseline_best_results = Baselines(collection_path).get_baseline_best_results()
        collection_name = collection_path.split('/')[-1]
        output_fn = os.path.join(self.all_results_root, 'tf3_compare', collection_name+'-compare_tf3_performances.csv')
        with open(output_fn, 'wb') as f:
            f.write('qid,tf3_MAP, best_baseline_MAP\n')
            for qid in performances_comparisons:
                f.write('%s,%f,%f\n' % (qid, performances_comparisons[qid][1], baseline_best_results[qid]) )



    def indri_run_query_atom(self, para_file):
        with open(para_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    row = line.split()
                    indri.IndriRunQuery(row[0], row[2], row[1])

    def run_tf3(self, collections_path=[]):
        frameinfo = getframeinfo(currentframe())
        current_function_name = inspect.stack()[0][3]
        for c in collections_path:
            collection_name = c.split('/')[-1] if c.split('/')[-1] else c.split('/')[-2]
            single_queries = Query(c).get_queries_of_length(1)
            qids = [ele['num'] for ele in single_queries]
            performances_comparisons = {}
            all_paras = []
            for i in xrange(1, 101):
                q_path = os.path.join(c, 'standard_queries')
                r_path = os.path.join(c, 'results', 'tf3_%d' % i)
                if not os.path.exists(r_path):
                    all_paras.append((q_path, '-rule=method:tf-3,f:%d'%i, r_path))
            if all_paras:
                #print all_paras
                MPI().gen_batch_framework(os.path.join(self.batch_root, collection_name, 'bin'), 
                    current_function_name, frameinfo.filename, '111', 
                    all_paras, 
                    os.path.join(self.batch_root, collection_name, 'misc', current_function_name), 
                    para_alreay_split=False,
                    add_node_to_para=False,
                    run_after_gen=True,
                    memory='4G'
                )
            else:
                print 'Nothing to RUN for '+c        

    def compare_tf3_performances(self, collections_path=[]):
        for c in collections_path:
            single_queries = Query(c).get_queries_of_length(1)
            qids = [ele['num'] for ele in single_queries]
            performances_comparisons = {}
            for i in xrange(1, 101):
                r_path = os.path.join(c, 'results', 'tf3_%d' % i)
                performances = Evaluation(c, r_path).get_all_performance_of_some_queries(qids=qids, return_all_metrics=False)
                for qid in performances:
                    if qid not in performances_comparisons:
                        performances_comparisons[qid] = {}
                    performances_comparisons[qid][i] = performances[qid]['map']
            # plot the per-query performance of TF3.
            # this may not be meaningful for single-term queries since 
            # the performances do not change for different fade points.
            # So for single-term queries, just compare it with the best results of baselines
            
            #self.plot_compare_tf3_performances(performances_comparisons, c)
            self.compare_tf3_with_baselines(performances_comparisons, c)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-111', '--indri_run_query_atom', nargs=1,
                       help='indri_run_query_atom')

    parser.add_argument('-1', '--plot_cost_of_rel_docs', action='store_true',
                       help='plot the cost of retrieving relevant documents')

    parser.add_argument('-2', '--batch_run_okapi_pivoted_without_idf', action='store_true',
                       help='batch run okapi pivoted without idf')

    parser.add_argument('-3', '--plot_single', nargs=3,
                       help='plot single figure')

    parser.add_argument('-4', '--plot_tfc_constraints', nargs='+',
                       help='plot the relevant document distribution. \
                       Take TF as an example and we start from the simplest case when \
                       |Q|=1.  We could leverage an existing collection \
                       and estimate P( c(t,D)=x | D is a relevant document), \
                       where x = 0,1,2,...maxTF(t). ')

    parser.add_argument('-51', '--run_tf3', nargs='+',
                       help='Batch run TF3 results')

    parser.add_argument('-52', '--compare_tf3_performances', nargs='+',
                       help='Compare all the performances of tf3 results. \
                       TF3: if tf<fade_point: score=tf; if tf>fade_point: score=log(tf).')

    args = parser.parse_args()

    if args.indri_run_query_atom:
        SingleQueryAnalysis().indri_run_query_atom(args.indri_run_query_atom[0])

    if args.plot_cost_of_rel_docs:
        SingleQueryAnalysis().plot_cost_of_rel_docs()

    if args.batch_run_okapi_pivoted_without_idf:
        SingleQueryAnalysis().batch_run_okapi_pivoted_without_idf()

    if args.plot_single:
        SingleQueryAnalysis().plot_single(args.plot_single[0], 
            args.plot_single[1], args.plot_single[2])

    if args.plot_tfc_constraints:
        SingleQueryAnalysis().plot_tfc_constraints(args.plot_tfc_constraints)

    if args.run_tf3:
        SingleQueryAnalysis().run_tf3(args.run_tf3)

    if args.compare_tf3_performances:
        SingleQueryAnalysis().compare_tf3_performances(args.compare_tf3_performances)
