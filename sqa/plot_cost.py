from single_query_analysis import SingleQueryAnalysis

class PlotCost(SingleQueryAnalysis):
    """
    Plot the cost of retrieving a relevant document
    """
    def __init__(self):
        super(PlotCost, self).__init__()

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

