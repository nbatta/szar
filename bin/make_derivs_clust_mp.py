#Python 2 compatibility stuff
from __future__ import print_function
from __future__ import division
from builtins import str
from builtins import range
from six.moves import configparser
import six
if six.PY2:
  ConfigParser = configparser.SafeConfigParser
else:
  ConfigParser = configparser.ConfigParser
#

from szar.clustering import Clustering
from szar.counts import ClusterCosmology
from szar.szproperties import SZ_Cluster_Model
import szar.fisher as sfisher
import numpy as np

import sys
import pickle as pickle

from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool 

class Derivs_Clustering(object):
    def __init__(self):
        self.params = None
        self.step = 1
        self.poolsize = 2

    def instantiate_params(self, config):
        manualParamList = config.get('general','manualParams').split(',')

        paramList = []
        fparams = {}
        stepSizes = {}
        for (key, val) in config.items('params'):
            if key in manualParamList: continue
            if ',' in val:
                param, step = val.split(',')
                paramList.append(key)
                fparams[key] = float(param)
                stepSizes[key] = float(step)
            else:
                fparams[key] = float(val)
        self.params = []
    
    def pspec(self, params):
        pass

    def veffective(self, params):
        pass

    def make_derivs(self):
        pool = ThreadPool(self.poolsize)

        ps_list = pool.map(self.pspec, self.param_gradients)

        pool.close()
        pool.join()

        veffs = self.veffective(self.params)

        fisher_factors = (ps**2 * self.ks**2 * self.deltaks * self.deltamus) / (8 * np.pi) 

        derivatives = (ps_ups - ps_downs) / (2 * self.step)

        return derivatives, fisher_factors


if __name__ == '__main__':
    INIFILE = "input/pipeline.ini"
    config = ConfigParser()
    config.optionxform=str
    config.read(INIFILE)

    deriv = Derivs_Clustering()
    deriv.instantiate_params(config)

    #fish_derivs, fish_facs = deriv.make_derivs()
