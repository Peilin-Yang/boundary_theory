import sys,os
import math
import argparse
import json
import ast
import subprocess
from subprocess import Popen, PIPE
from datetime import datetime
from operator import itemgetter
import multiprocessing
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class BasicPlot(object):
    """docstring for BasicPlot"""
    def __init__(self, **arg):
        super(BasicPlot, self).__init__()
        self.arg = arg

    def plot_tf_ln_relation(self):
        """
        Plot the relationship of TF and LN for current function

        @Input:
            None
        @Output:
            matplotlib figure
        @Return: 
            None
        """
        raise NotImplementedError("Please Implement this method")


class BM25(BasicPlot):
    def __init__(self, **arg):
        super(BM25, self).__init__()
        self.arg = arg

        self.k1 = 1.2
        self.b = 0.75

    def plot_tf_ln_relation(self):
        x = np.arange(1., 50., 1)
        #for ln in np.arange(int(self.arg['avdl']/10), int(self.arg['avdl']*10), 100):
        ln = self.arg['avdl']/10
        y = self.k1*x/(self.k1*(1-self.b+self.b*(ln+x)/self.arg['avdl'])+x)
        plt.plot(x, y, linewidth=2.0)
        plt.savefig('BM25_tf_ln.png', format='png', bbox_inches='tight')


class Pivoted(BasicPlot):
    def __init__(self, **arg):
        super(Pivoted, self).__init__()
        self.arg = arg

        self.s = 0.4

    def plot_tf_ln_relation(self):
        x = np.arange(1., 50., 1)
        #for ln in np.arange(int(self.arg['avdl']/10), int(self.arg['avdl']*10), 100):
        ln = self.arg['avdl']
        y = (1+np.log(1+np.log(x)))/(1-self.s+self.s*(ln+x)/self.arg['avdl'])
        plt.plot(x, y, linewidth=2.0)
        plt.savefig('Pivoted_tf_ln.png', format='png', bbox_inches='tight')



if __name__ == '__main__':
    BM25(**{'avdl':500}).plot_tf_ln_relation()
    #Pivoted(**{'avdl':500}).plot_tf_ln_relation()
