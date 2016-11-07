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


class EM(object):
    """
    Expectation Maximization algorithm
    """

    def __init__(self):
        super(EM, self).__init__()

    def exponential(self, data=[], init_lambdas=[2,0.5], max_iteration=50):
        """
        two mixture of exponential
        """
        data = np.asarray(data)
        idx = 1
        lambdas = np.asarray(init_lambdas)
        while idx < max_iteration:
            x = [lmbda*np.exp(data*lmbda) for lmbda in lambdas]
            weights = x/np.sum(x, axis=0)
            coefficients = np.mean(weights, axis=1)
            print x, weights, coefficients, data
            lambdas = len(data)*1./np.sum(weights*data, axis=1)
            print lambdas
            raw_input()
            idx+=1
        print weights

class Test(unittest.TestCase):
    pass

if __name__ == '__main__':
    #unittest.main()
    em = EM()
    a = [70,40,20,10,9,8,7,6,5,4,3,2,2,2,2,2,2,1,1,1,1,1]
    em.exponential(np.asarray(a)*1./np.sum(a))
