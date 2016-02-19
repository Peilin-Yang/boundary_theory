from single_query_analysis import SingleQueryAnalysis

class PlotTFRel(SingleQueryAnalysis):
    """
    Plot the probability distribution of P(TF=x|D=1) and P(D=1|TF=x)
    """

    def plot_single_tfc_constraints_draw_pdf(self, ax, xaxis, yaxis, 
            title, legend, legend_outside=False, marker='ro', 
            xlog=True, ylog=False, zoom=False):
        # 1. probability distribution 
        ax.plot(xaxis, yaxis, marker, ms=4, label=legend)
        ax.vlines(xaxis, [0], yaxis)
        if xlog:
            ax.set_xscale('log')
        if ylog:
            ax.set_yscale('log')
        ax.set_xlim(0, ax.get_xlim()[1] if ax.get_xlim()[1]<100 else 100)
        #ax.set_ylim(0, ax.get_ylim()[1] if ax.get_ylim()[1]<500 else 500)
        ax.set_title(title)
        ax.legend(loc='upper right')

        # zoom
        if zoom:
            axins = inset_axes(ax,
                   width="50%",  # width = 30% of parent_bbox
                   height=0.8,  # height : 1 inch
                   loc=7) # center right
            zoom_xaxis = []
            for x in xaxis:
                if x <= 20:
                    zoom_xaxis.append(x)
            zoom_yaxis = yaxis[:len(zoom_xaxis)]
            axins.plot(zoom_xaxis, zoom_yaxis, marker, ms=4)
            axins.vlines(zoom_xaxis, [0], zoom_yaxis)
            axins.set_xlim(0, 20)
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")


    def plot_single_tfc_constraints_draw_kde(self, ax, yaxis, _bandwidth=0.5):
        # kernel density estimation 
        #print '_bandwidth:'+str(_bandwidth)
        yaxis = np.asarray(yaxis)[:, np.newaxis]
        kde = KernelDensity(kernel='gaussian', bandwidth=_bandwidth).fit(yaxis)
        X_plot = np.linspace(0, len(yaxis), 100)[:, np.newaxis]
        log_dens = kde.score_samples(X_plot)
        ax.fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF',
                label='KDE')
        #print kde
        ax.legend(loc='best')

    def plot_single_tfc_constraints_draw_hist(self, ax, yaxis, nbins, _norm, title, legend):
        #2. hist gram
        yaxis.sort()
        n, bins, patches = ax.hist(yaxis, nbins, normed=_norm, facecolor='#F08080', alpha=0.5, label=legend)
        ax.set_title(title)
        ax.legend()


    def plot_single_tfc_constraints_tf_rel(self, collection_path, smoothing=True, oformat='eps'):
        collection_name = collection_path.split('/')[-1]
        cs = CollectionStats(collection_path)
        output_root = 'single_query_figures'
        single_queries = Query(collection_path).get_queries_of_length(1)
        queries = {ele['num']:ele['title'] for ele in single_queries}
        #print queries
        rel_docs = Judgment(collection_path).get_relevant_docs_of_some_queries(queries.keys(), 1, 'dict')
        #print rel_docs
        #raw_input()
        collection_level_tfs = []
        collection_level_x_dict = {}
        collection_level_maxTF = 0
        num_cols = 4
        num_rows = int(math.ceil(len(rel_docs)*1.0/num_cols*2))
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, sharex=False, sharey=False, figsize=(3*num_cols, 3.*num_rows))
        font = {'size' : 8}
        plt.rc('font', **font)
        row_idx = 0
        col_idx = 0
        for qid in sorted(rel_docs):
            ax1 = axs[row_idx][col_idx]
            ax2 = axs[row_idx][col_idx+1]
            col_idx += 2
            if col_idx >= num_cols:
                row_idx += 1
                col_idx = 0
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
            yaxis_all = []
            for tf in range(0, maxTF+1):
                if tf not in x_dict:
                    x_dict[tf] = 0
                else:
                    yaxis_all.extend([tf+.1]*x_dict[tf])
                if smoothing:
                    x_dict[tf] += .1
                    rel_docs_len += .1

                if tf not in collection_level_x_dict:
                    collection_level_x_dict[tf] = 0
                if smoothing:
                    collection_level_x_dict[tf] += .1

            xaxis = x_dict.keys()
            xaxis.sort()
            yaxis_pdf = [x_dict[x]/rel_docs_len for x in xaxis]

            self.plot_single_tfc_constraints_draw_pdf(ax1, xaxis, 
                yaxis_pdf, qid+'-'+query_term, 
                "maxTF=%d\n|rel_docs|=%d\nidf=%.1f" % (maxTF, rel_docs_len, idf), 
                xlog=False)
            self.plot_single_tfc_constraints_draw_kde(ax1, yaxis_all, 1.06*math.pow(len(yaxis_all), -0.2)*np.std(yaxis_all))
            self.plot_single_tfc_constraints_draw_hist(ax2, yaxis_hist, 
                math.ceil(maxTF/10.), False, qid+'-'+query_term, 
                '#bins(maxTF/10.0)=%d' % (math.ceil(maxTF/10.)))

        fig.text(0.5, 0.07, 'Term Frequency', ha='center', va='center', fontsize=12)
        fig.text(0.06, 0.5, 'P( c(t,D)=x | D is a relevant document)=tf/|rel_docs|', ha='center', va='center', rotation='vertical', fontsize=12)
        fig.text(0.5, 0.5, 'Histgram', ha='center', va='center', rotation='vertical', fontsize=12)
        fig.text(0.6, 0.04, 'Histgram:rel docs are binned by their TFs. The length of the bin is set to 10. Y axis shows the number of rel docs in each bin.', ha='center', va='center', fontsize=10)

        plt.savefig(os.path.join(self.all_results_root, output_root, collection_name+'-tf_rel.'+oformat), 
            format=oformat, bbox_inches='tight', dpi=400)

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

        plt.savefig(os.path.join(self.all_results_root, output_root, collection_name+'-all-tf_rel.'+oformat), 
            format=oformat, bbox_inches='tight', dpi=400)



    def plot_single_tfc_constraints_rel_tf(self, collection_path, plot_ratio=True, smoothing=False, oformat='eps'):
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

        num_cols = 4
        num_rows = int(math.ceil(len(rel_docs)*1.0/num_cols))
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, sharex=False, sharey=False, figsize=(3*num_cols, 3.*num_rows))
        font = {'size' : 8}
        plt.rc('font', **font)
        row_idx = 0
        col_idx = 0
        for qid in sorted(rel_docs):
            ax = axs[row_idx][col_idx]
            col_idx += 1
            if col_idx >= num_cols:
                row_idx += 1
                col_idx = 0
            query_term = queries[qid]
            maxTF = cs.get_term_maxTF(query_term)
            idf = cs.get_term_IDF1(query_term)
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
            yaxis_all = []
            for row in cs.get_qid_details(qid):
                qid_docs_len += 1
                tf = int(row['total_tf'])
                rel = (int(row['rel_score'])>=1)
                if tf not in x_dict:
                    x_dict[tf] = [0, 0] # [rel_docs, total_docs]
                if rel:
                    x_dict[tf][0] += 1
                x_dict[tf][1] += 1
                yaxis_all.append(tf)

                if tf not in collection_level_x_dict:
                    collection_level_x_dict[tf] = [0, 0] # [rel_docs, total_docs]
                if rel:
                    collection_level_x_dict[tf][0] += 1
                collection_level_x_dict[tf][1] += 1

            xaxis = x_dict.keys()
            xaxis.sort()
            yaxis = [x_dict[x][0] for x in xaxis]
            yaxis_total = [x_dict[x][1] for x in xaxis]
            yaxis_ratio = [x_dict[x][0]/x_dict[x][1] for x in xaxis]
            #print xaxis
            #print yaxis
            xaxis_splits_10 = [[x for x in xaxis if x <= i+10 and x > i] for i in range(0, maxTF+1, 10)]
            #print xaxis_splits_10
            yaxis_splits_10 = [[x_dict[x][0]*1./x_dict[x][1] for x in xaxis if x <= i+10 and x > i] for i in range(0, maxTF+1, 10)]
            #print yaxis_splits_10
            entropy_splits_10 = [entropy(ele, base=2) for ele in yaxis_splits_10]
            query_stat = cs.get_term_stats(query_term)
            dist_entropy = entropy(yaxis, base=2)
            if plot_ratio:
                self.plot_single_tfc_constraints_draw_pdf(ax, xaxis, 
                    yaxis_ratio, qid+'-'+query_term, 
                    'total\nidf:%.1f'%idf,
                    True,
                    xlog=False)
            else:
                self.plot_single_tfc_constraints_draw_pdf(ax, xaxis, yaxis_total, 
                    qid+'-'+query_term, 
                    'total\nidf:%.1f'%idf, 
                    True,
                    xlog=False)
                self.plot_single_tfc_constraints_draw_pdf(ax, xaxis, 
                    yaxis, qid+'-'+query_term, 
                    'rel', 
                    True,
                    marker='bs', 
                    xlog=False,
                    zoom=(qid =='379' or qid =='395' or qid =='417' or qid =='424'))


        collection_vocablulary_stat = cs.get_vocabulary_stats()
        collection_vocablulary_stat_str = ''
        idx = 1
        for k,v in collection_vocablulary_stat.items():
            collection_vocablulary_stat_str += k+'='+'%.2f'%v+' '
            if idx == 3:
                collection_vocablulary_stat_str += '\n'
                idx = 1
            idx += 1
        #fig.text(0.5, 0, collection_vocablulary_stat_str, ha='center', va='center', fontsize=12)
        if plot_ratio:
            output_fn = os.path.join(self.all_results_root, output_root, collection_name+'-rel_tf_ratio.'+oformat)
        else:
            os.path.join(self.all_results_root, output_root, collection_name+'-rel_tf.'+oformat)
        plt.savefig(output_fn, format=oformat, bbox_inches='tight', dpi=400)


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
        plt.savefig(os.path.join(self.all_results_root, output_root, collection_name+'-all-rel_tf.'+oformat), 
            format=oformat, bbox_inches='tight', dpi=400)


    def plot_single_tfc_constraints(self, corpus_path):
        #self.plot_single_tfc_constraints_tf_rel(corpus_path)
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
