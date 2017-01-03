# -*- coding: utf-8 -*-
import sys,os
import json
import re
import string
import ast
import xml.etree.ElementTree as ET
import tempfile
from subprocess import Popen, PIPE
from inspect import currentframe, getframeinfo
import argparse

class Query(object):
    """
    Get the judgments of a corpus.
    When constructing, pass the path of the corpus. For example, "../wt2g/"
    """
    def __init__(self, path):
        self.corpus_path = os.path.abspath(path)
        if not os.path.exists(self.corpus_path):
            frameinfo = getframeinfo(currentframe())
            print frameinfo.filename, frameinfo.lineno
            print '[Query Constructor]:path "' + self.corpus_path + '" is not a valid path'
            print '[Query Constructor]:Please provide a valid corpus path'
            exit(1)

        self.query_file_path = os.path.join(self.corpus_path, 'raw_topics')
        if not os.path.exists(self.query_file_path):
            frameinfo = getframeinfo(currentframe())
            print frameinfo.filename, frameinfo.lineno
            print """No query file found! 
                query file should be called "raw_topics" under 
                corpus path. You can create a symlink for it"""
            exit(1)

        self.parsed_query_file_path = os.path.join(self.corpus_path, 'parsed_topics.json')


    def write_query_file(self, t=[]):
        fpath = 'jsdf9HNOJKh90dfjflsdf'
        with open(fpath, 'w') as f:
            for ele in t:
                f.write('<DOC>\n')
                f.write('<TEXT>\n')
                f.write(ele)
                f.write('\n')
                f.write('</TEXT>\n')
                f.write('</DOC>\n')
        return fpath


    def parse_query(self, t=[]):
        """
        use IndriTextTransformer to parse the query
        """
        fpath = self.write_query_file(t)
        try:
            process = Popen(['IndriTextTransformer', '-class=trectext', '-file='+fpath], stdout=PIPE)
            stdout, stderr = process.communicate()
            r = []
            for line in stdout.split('\n'):
                line = line.strip()
                if line:
                    r.append(line)
            os.remove(fpath)
        except:
            os.remove(fpath)
            raise NameError("parse query error!")
        return r


    def get_queries(self):
        """
        Get the query of a corpus

        @Return: a list of dict [{'num':'401', 'title':'the query terms',
         'desc':description, 'narr': narrative description}, ...]
        """

        if not os.path.exists(self.parsed_query_file_path):
            with open(self.query_file_path) as f:
                s = f.read()
                all_topics = re.findall(r'<top>.*?<\/top>', s, re.DOTALL)
                #print all_topics
                #print len(all_topics)

                _all = []
                for t in all_topics:
                    t = re.sub(r'<\/.*?>', r'', t, flags=re.DOTALL)
                    a = re.split(r'(<.*?>)', t.replace('<top>',''), re.DOTALL)
                    #print a
                    aa = [ele.strip() for ele in a if ele.strip()]
                    d = {}
                    for i in range(0, len(aa), 2):
                        """
                        if i%2 != 0:
                            if aa[i-1] == '<num>':
                                aa[i] = aa[i].split()[1]
                            d[aa[i-1][1:-1]] = aa[i].strip().replace('\n', ' ')
                        """
                        tag = aa[i][1:-1]
                        value = aa[i+1].replace('\n', ' ').strip().split(':')[-1].strip()
                        if tag != 'num' and value:
                            value = self.parse_query([value])[0]
                        if tag == 'num':
                            value = str(int(value)) # remove the trailing '0' at the beginning
                        d[tag] = value
                    _all.append(d)

            with open(self.parsed_query_file_path, 'wb') as f:
                json.dump(_all, f, indent=2)

        with open(self.parsed_query_file_path) as f:
            return json.load(f)

    def get_queries_dict(self):
        """
        Get the query of a corpus

        @Return: a dict with keys as qids {'401':{'title':'the title', 'desc':'the desc'}, ...}
        """
        all_queries = self.get_queries()
        all_queries_dict = {}
        for ele in all_queries:
            qid = ele['num']
            all_queries_dict[qid] = ele

        return all_queries_dict
        

    def print_query_len_dist(self, part='title'):
        queries = self.get_queries()
        lens = {}
        for q in queries:
            l = len(q[part].split())
            if l not in lens:
                lens[l] = 0
            lens[l] += 1

        print '| | |'
        print '|---|---|---|'
        for k in sorted(lens):
            print '| %d | %d | %.1f%% |' % (k, lens[k], lens[k]*100.0/len(queries))


    def get_queries_lengths(self, part='title'):
        """
        For a set of queries, return the lengths of the queries

        @Return: a list of integers showing the lengths of the queries
        """
        queries = self.get_queries()
        lengths = set([len(q[part].split()) for q in queries])
        lengths = list(lengths)
        lengths.sort()
        return lengths


    def get_queries_of_length(self, length, part='title'):
        """
        Get the queries of a specific length

        @Input:
            length - the specific length. For example, length=1 get all queries
                     with single term

        @Return: a list of dict [{'num':'403', 'title':'osteoporosis',
         'desc':description, 'narr': narrative description}, ...]
        """

        all_queries = self.get_queries()
        filtered_queries = [ele for ele in all_queries if len(ele[part].split()) == length]

        return filtered_queries


    def indent(self, elem, level=0):
        i = "\n" + level*"  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self.indent(elem, level+1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i


    def gen_query_file_for_indri(self, output_path='standard_queries', 
            index_path='index', is_trec_format=True, count=9999999):
        """
        generate the query file for Indri use.

        @Input:
            output_path - the path to output query file, default "standard_queries"
            index_path - the index path, default "index".
            is_trec_format - whether to output the results in TREC format, default True
            count - how many documents will be returned for each topic, default 9999999
        """
        all_topics = self.get_queries()

        qf = ET.Element('parameters')
        index = ET.SubElement(qf, 'index')
        index.text = os.path.join(self.corpus_path, index_path)
        ele_trec_format = ET.SubElement(qf, 'trecFormat')
        ele_trec_format.text = 'true' if is_trec_format else 'false'
        ele_count = ET.SubElement(qf, 'count')
        ele_count.text = str(count)
        for ele in all_topics:
            t = ET.SubElement(qf, 'query')
            qid = ET.SubElement(t, 'number')
            qid.text = ele['num']
            q = ET.SubElement(t, 'text')
            q.text = ele['title']

        self.indent(qf)

        tree = ET.ElementTree(qf)
        tree.write(os.path.join(self.corpus_path, output_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-1", "--gen_standard_queries",
        nargs=1,
        help="Generate the standard queries for Indri. Please give the collection path!")

    parser.add_argument("-2", "--print_query_len_dist",
        nargs=1,
        help="Print the distribution of query lengths. Please give the collection path!")


    args = parser.parse_args()

    if args.gen_standard_queries:
        Query(args.gen_standard_queries[0]).gen_query_file_for_indri()

    if args.print_query_len_dist:
        Query(args.print_query_len_dist[0]).print_query_len_dist()

