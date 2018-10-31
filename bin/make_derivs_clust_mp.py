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
        self.params = np.array([])
        self.step = 1
        self.poolsize = 2

    def instantiate_params(self):
        pass
    
    def pspec(self, parameters):
        pass

    def veffective(self):
        pass

    def make_derivs(self):
        pool = ThreadPool(self.poolsize)

        ps = pool.map(self.pspec, self.params)
        veffs = pool.map(self.veffective, self.params)

        pool.close()
        pool.join()

        ps_ups = self.pspec(self.params + self.step)
        ps_downs = self.pspec(self.params - self.step)

        fisher_factors = (ps**2 * self.ks**2 * self.deltaks * self.deltamus) / (8 * np.pi) 

        derivatives = (ps_ups - ps_downs) / (2 * self.step)

        return derivatives, fisher_factors


if __name__ == '__main__':
    deriv = Derivs_Clustering()
    deriv.instantiate_params()

    fish_derivs, fish_facs = deriv.make_derivs()
