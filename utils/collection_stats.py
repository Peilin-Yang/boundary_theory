import os,sys
import csv
import ast
import json
import tempfile
import subprocess
from subprocess import Popen, PIPE


class CollectionStats(object):
    def __init__(self, path):
        self.collection_path = os.path.abspath(path)
        if not os.path.exists(self.collection_path):
            print '[CollectionStats Constructor]:Please provide a valid collection path'
            exit(1)

        self.fieldnames=['qid', 'docid', 'score', 'rel_score', 'tf', 'total_tf', 
            'doc_len', 'doc_minTF', 'doc_maxTF', 'doc_avgTF', 'doc_varTF']

    #index############################################################

    def get_index_statistics(self):
        process = Popen(['dumpindex', os.path.join(self.collection_path, 'index'), 's'], stdout=PIPE)
        stdout, stderr = process.communicate()
        stats = {}
        for line in stdout.split('\n'):
            line = line.strip()
            if line:
                row = line.split(':')
                try:
                    stats[row[0].strip()] = ast.literal_eval(row[1].strip())
                except:
                    continue

        return stats        

    def get_avdl(self):
        all_statistics = self.get_index_statistics()
        return all_statistics.get('average doc length', None)

    def get_doc_counts(self):
        all_statistics = self.get_index_statistics()
        return all_statistics.get('documents', None)


    #vocabulary############################################################
    def get_vocabulary_stats(self, feature=None):
        """
        Get collection level vocabulary statistics.
        This includes DF (Document Frequency) Related statistics
        and 
        TFC (Term Frequency in Collection) Related statistics.
        """
        path = os.path.join(self.collection_path, 'vocabulary_stats.json')
        if not os.path.exists(path):
            with open(path, 'wb') as f:
                subprocess.Popen(['dumpindex', 
                    os.path.join(self.collection_path, 'index'), 'vs'], 
                    stdout=f)
                f.flush()

        with open(path) as f:
            r = json.load(f)
        
        if feature:
            return r[feature]
        else:
            return r

    # Document Frequency Related
    def get_vocaulary_maxDF(self):
        return self.get_vocabulary_stats('maxDF')
    def get_vocaulary_minDF(self):
        return self.get_vocabulary_stats('minDF')
    def get_vocaulary_avgDF(self):
        return self.get_vocabulary_stats('avgDF')
    def get_vocaulary_varDF(self):
        return self.get_vocabulary_stats('varDF')

    # Term Frequency in Collection Related
    def get_vocaulary_maxTFC(self):
        return self.get_vocabulary_stats('maxTFC')
    def get_vocaulary_minTFC(self):
        return self.get_vocabulary_stats('minTFC')
    def get_vocaulary_avgTFC(self):
        return self.get_vocabulary_stats('avgTFC')
    def get_vocaulary_varTFC(self):
        return self.get_vocabulary_stats('varTFC')


    #document############################################################

    def get_document_stats(self, internal_docids):
        """
        Get a list of documents' statistics.
        We use the modified "dumpindex" indri code to do this in faster way.
        It is better to send a list of internal docids to Indri code so that 
        it can be faster!!!
        """
        flag, tmp_fn = tempfile.mkstemp(dir='.')
        with open(tmp_fn, 'wb') as f:
            f.write('\n'.join(internal_docids))

        process = Popen(['dumpindex', 
            os.path.join(self.collection_path, 'index'), 'ds', tmp_fn], 
            stdout=PIPE)
        stdout, stderr = process.communicate()

        all_doc_stats = {}
        for line in stdout.split('\n'):
            line = line.strip()
            if line:
                this = {}
                cur_did = None
                row = line.split(',')
                for ele in row:
                    k = ele.split(':')[0]
                    v = ele.split(':')[1]
                    if k == 'id':
                        cur_did = v
                    else:
                        this[ele.split(':')[0]] = ast.literal_eval(ele.split(':')[1])

                all_doc_stats[cur_did] = this

        os.remove(tmp_fn)
        return all_doc_stats



    #term############################################################
    def get_term_stats(self, term, feature=None):
        """
        Get a statistics of a term

        @Input:
            term (string) : the term that whose statistics is needed
            feature (string) : the required feature

        @Return: the required statistics
        """

        process = Popen(['dumpindex', 
            os.path.join(self.collection_path, 'index'), 'termfeature', term], 
            stdout=PIPE)
        stdout, stderr = process.communicate()
        r = json.loads(stdout)
        if feature:
            return r[feature]
        else:
            return r

    
    def get_term_stem(self, term):
        return self.get_term_stats(term, 'stem')

    def get_term_df(self, term):
        """
        document frequency
        """
        return self.get_term_stats(term, 'df')

    def get_term_IDF1(self, term):
        """
        N/df
        """
        return self.get_term_stats(term, 'idf1')

    def get_term_maxTF(self, term):
        return self.get_term_stats(term, 'maxTF')

    def get_term_minTF(self, term):
        return self.get_term_stats(term, 'minTF')

    def get_term_avgTF(self, term):
        return self.get_term_stats(term, 'avgTF')

    def get_term_varTF(self, term):
        return self.get_term_stats(term, 'varTF')


    ############################################################################

    def get_internal_docid(self, docno):
        """
        Get the internal docid of a document with docno

        @Input:
            docno (string) : the docno which is shown in result file

        @Return: the internal docid
        """

        process = Popen(['dumpindex', 
            os.path.join(self.collection_path, 'index'), 'di', 'docno', docno], 
            stdout=PIPE)
        stdout, stderr = process.communicate()

        return stdout.strip()

    def get_external_docid(self, docid):
        """
        Get the external docid of a document with docid

        @Input:
            docid (string) : the internal docid

        @Return: the external docid
        """

        process = Popen(['dumpindex', 
            os.path.join(self.collection_path, 'index'), 'dn', docid], stdout=PIPE)
        stdout, stderr = process.communicate()

        return stdout.strip()


    def get_docs_tf_of_term(self, term, docs=[]):
        """
        Get the term frequency of the term for a list of documents

        @Input:
            term (string) : the query term
            docs (list) : a list of documents which are essentially the external ids

        @Return: tf value list (list of doubles)
        """
        r = []
        term_ivlist = self.get_term_counts_dict(term)
        for docno in docs:
            internal_docid = self.get_internal_docid(docno)
            r.append(term_ivlist[internal_docid][0]) if internal_docid in term_ivlist else 0
        return r

    def get_qid_details(self, qid):
        with open(os.path.join(self.collection_path, 'detailed_doc_stats', qid)) as f:
            rows = csv.DictReader(f, fieldnames=self.fieldnames)
            for row in rows:
                yield row

    def get_term_docs_tf_of_term_with_qid(self, qid, term, docs=[]):
        """
        Get the term frequency of the term for a list of documents

        @Input:
            qid (string) : the query id of which term occurs (see below)
            term (string) : the query term
            docs (list) : a list of documents which are essentially the external ids

        @Return: tf value list (list of doubles)
        """
        r = []
        for row in self.get_qid_details(qid):
            if row['docid'] in docs:
                found = False
                for ele in row['tf'].split(','):
                    if ele.split('-')[0].lower() == term:
                        r.append(ast.literal_eval(ele.split('-')[1]))
                        found = True
                        break
                if not found:
                    r.append(0)
        return r

    def get_richStats(self):
        richStatsFilePath = os.path.join(self.collection_path, 'rich_stats.json')
        if not os.path.exists(richStatsFilePath):
            f = open(richStatsFilePath, 'wb')
            all_performances = {}
            process = Popen(['dumpindex', os.path.join(self.collection_path, 'index'), 'rs'], stdout=f)
            stdout, stderr = process.communicate()
            f.close()

        with open(richStatsFilePath) as f:
            return json.load(f)

    def get_term_counts(self, term):
        process = Popen(['dumpindex', 
            os.path.join(self.collection_path, 'index'), 't', term], stdout=PIPE)
        stdout, stderr = process.communicate()

        all_term_counts = []
        for line in stdout.split('\n')[1:-2]:
            line = line.strip()
            if line:
                row = line.split()
                all_term_counts.append(row)

        return all_term_counts

    def get_term_counts_dict(self, term):
        process = Popen(['dumpindex', 
            os.path.join(self.collection_path, 'index'), 't', term], stdout=PIPE)
        stdout, stderr = process.communicate()

        all_term_counts = {}
        for line in stdout.split('\n')[1:-2]:
            line = line.strip()
            if line:
                row = line.split()
                all_term_counts[row[0]] = [row[1], row[2]]

        return all_term_counts


if __name__ == '__main__':
    # CollectionStats('../../wt2g/').re_gen_do_statistics_json()
    # CollectionStats('../../trec8/').re_gen_do_statistics_json()
    # CollectionStats('../../trec7/').re_gen_do_statistics_json()
    CollectionStats(sys.argv[1]).get_vocabulary_stats("varDF")
