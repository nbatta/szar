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
from orphics.io import dict_from_section, list_from_config
import numpy as np

import sys
import pickle as pickle

from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool 

class Derivs_Clustering(object):
    def __init__(self, config):
        self.params = None
        self.poolsize = 2
        self.config = config

    def instantiate_params(self):
        manual_param_list = self.config.get('general','manualParams').split(',')

        paramlist = []
        paramdict = {}
        stepsizes = {}
        for (key, val) in self.config.items('params'):
            if key in manual_param_list:
                continue
            if ',' in val:
                param, step = val.split(',')
                paramlist.append(key)
                paramdict[key] = float(param)
                stepsizes[key] = float(step)/2
            else:
                paramdict[key] = float(val)
                stepsizes[key] = 0
        
        param_ups = []
        param_downs = []
        for key,val in paramdict.items():
            param_up = paramdict.copy()
            param_down = paramdict.copy()

            param_up[key] = val + stepsizes[key]
            param_down[key] = val - stepsizes[key]

            param_ups.append(param_up)
            param_downs.append(param_down)

        self.params = paramdict
        self.param_ups = param_ups
        self.param_downs = param_downs

    def instantiate_grids(self):
        mubins = list_from_config(self.config,'general','mubins')
        self.mus = np.linspace(mubins[0],mubins[1],int(mubins[2])+1)

        clst = Clustering()
        self.ks = clst.HMF.kh
        self.ps_fid = clst.fine_ps_bar(self.mus)
        self.veff_fid = clst.V_eff(self.mus)

        self.deltaks = np.gradient(ks)
        self.deltamus =np.gradient(mus)

    def _pspec(self, params):
        constdict = dict_from_section(self.config,'constants')
        clttfile = self.config.get('general','clttfile')
        cc = ClusterCosmology(paramDict=params, constDict=constdict, clTTFixFix=clttfile)
        clst = Clustering(self.exp_name, self.grid_name, self.version, cc)
        return clst.fine_ps_bar(self.mus)

    def make_derivs(self):
        pool = Pool(self.poolsize) #Can also make a ThreadPool

        ps_ups = pool.map(self._pspec, self.param_ups)
        ps_downs = pool.map(self._pspec, self.param_downs)

        pool.close()
        pool.join()

        fisher_factors = (self.ps_fid**2 * self.veff_fid * self.ks**2 * self.deltaks * self.deltamus) / (8 * np.pi) 

        derivatives = (ps_ups - ps_downs) / (2 * self.step)

        return derivatives, fisher_factors


if __name__ == '__main__':
    INIFILE = "input/pipeline.ini"
    config = ConfigParser()
    config.optionxform=str
    config.read(INIFILE)

    deriv = Derivs_Clustering(config)
    deriv.instantiate_params()
    deriv.instantiate_grids()

    #fish_derivs, fish_facs = deriv.make_derivs()
