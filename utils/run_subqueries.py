# -*- coding: utf-8 -*-
import sys,os
import json
import re
import string
import ast
import xml.etree.ElementTree as ET
import itertools
from subprocess import Popen, PIPE
from inspect import currentframe, getframeinfo
import argparse
from performance import Performances

class RunSubqueries(object):
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

    def get_queries(self):
        """
        Get the query of a corpus

        @Return: a list of dict [{'num':'401', 'title':'the query terms',
         'desc':description, 'narr': narrative description}, ...]
        """
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

    def run_indri_runquery(self, query_str, qid='0', rule='', index_path='index', count=1000):
        p = Popen(['IndriRunQuery_EX -index=%s -trecFormat=True -count=%d -query.number=%s -query.text="%s" -rule=%s' 
            % (os.path.join(corpus_path, 'index'), count, qid, query_str, rule)], bash=True, stdout=PIPE, stderr=PIPE)
        returncode = p.wait()
        stdout, stderr = p.communicate()
        if returncode == 0:
            return stdout
        else:
            print stdout, stderr
            exit()

    def get_subqueries(self, query_str):
        all_subqueries = {}
        terms = query_str.split()
        for i in range(1, len(terms)):
            j = 0
            for ele in itertools.combinations(terms, i):
                all_subqueries[' '.join(ele)] = '%d%d' % (i, j)
                j += 1
        return all_subqueries

    def batch_run_subqueries_paras(self, query_length=0):
        if query_length == 0: #all queries
            queries = self.get_queries()
        else:
            queries = self.get_queries_of_length(query_length)
        queries = {ele['num']:ele['title'] for ele in queries}
        for qid, query in queries.items():
            print qid, self.get_subqueries(query)
            raw_input()

