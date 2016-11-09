# -*- coding: utf-8 -*-
import sys,os
import math
import re
import argparse
import json
import ast
import copy
from subprocess import Popen, PIPE
from operator import itemgetter

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils/'))
from collection_stats import CollectionStats
from query import Query
from judgment import Judgment
from evaluation import Evaluation
from performance import Performances
from gen_doc_details import GenDocDetails

import numpy as np
import scipy.stats

import unittest


class Utils(object):
    """
    Some Utils
    """

    def __init__(self):
        super(Utils, self).__init__()

    
    def A(self, pr, pn, r, n):
        """
        """
        if r == 0:
           return 0
        R = {}
        for i in range(1, r+1):
            for j in range(n+1):
                R[(pr+r-i, pn+n-j, i, 0)] = (pr+r-i+1.0) / (pr+r-i+pn+n-j+1.0) 
                if i != 1:
                    R[(pr+r-i, pn+n-j, i, 0)] += R[(pr+r-i+1, pn+n-j, i-1, 0)]
        for i in range(1, r+1):
            for j in range(1, n+1):
                subR = R[(pr+r-i+1, pn+n-j, i-1, j)] if i!=1 else 0
                prob_r = i*1.0/(i+j)*((pr+r-i+1.0)/(pr+r-i+pn+n-j+1)+subR) 
                prob_n = j*1.0/(i+j)*R[(pr+r-i, pn+n-j+1, i, j-1)]
                R[(pr+r-i, pn+n-j, i, j)] = prob_r + prob_n
        return R[(pr, pn, r, n)]


    def cal_expected_map(self, ranking_list, total_rel=0):
        """
        Calculate the MAP based on the ranking_list.

        Input:
        @ranking_list: The format of the ranking_list is:
            [(num_rel_docs, num_total_docs), (num_rel_docs, num_total_docs), ...]
            where the index corresponds to the TF, e.g. ranking_list[1] is TF=1
        """
        s = 0.0
        pr = 0
        pn = 0
        for ele in reversed(ranking_list):
            rel_doc_cnt = ele[0]
            this_doc_cnt = ele[1]
            nonrel_doc_cnt = this_doc_cnt - rel_doc_cnt
            s += self.A(pr, pn, rel_doc_cnt, nonrel_doc_cnt)
            pr += rel_doc_cnt
            pn += nonrel_doc_cnt
            total_rel += rel_doc_cnt
        #print s/total_rel
        if total_rel == 0:
            return 0
        return s/total_rel


class Test(unittest.TestCase):
    def test_A(self):
        self.assertEqual(round(PlotSyntheticMAP().A(0,0,1,0), 3), 1.000)
        self.assertEqual(round(PlotSyntheticMAP().A(1,0,0,1), 3), 0.000)
        self.assertEqual(round(PlotSyntheticMAP().A(1,1,1,1), 3), 0.583)


if __name__ == '__main__':
    unittest.main()
