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

#cosmo imports
from szar.clustering import Clustering
from szar.counts import ClusterCosmology
from szar.szproperties import SZ_Cluster_Model
import szar.fisher as sfisher
from orphics.io import dict_from_section, list_from_config
import numpy as np

# etc
import sys
import pickle as pickle
import argparse

class Derivs_Clustering(object):
    def __init__(self, inifile):
        self.params = None
        self.inifile = inifile
        self.config = ConfigParser()
        self.config.optionxform=str
        self.config.read(self.inifile)

        self.constdict = dict_from_section(self.config,'constants')
        self.clttfile = self.config.get('general','clttfile')

        parser = argparse.ArgumentParser()
        parser.add_argument("expname", help="name of experiment")
        parser.add_argument("gridname", help="name of grid")
        parser.add_argument("version", help="version number")
        args = parser.parse_args()

        self.expname = args.expname
        self.gridname = args.gridname
        self.version = args.version

        self.saveid = self.expname + "_" + self.gridname + "_v" + self.version

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
        self.steps = np.fromiter(stepsizes.values(), dtype=float)

    def instantiate_grids(self):
        mubins = list_from_config(self.config,'general','mubins')
        self.mus = np.linspace(mubins[0],mubins[1],int(mubins[2])+1)

        cc = ClusterCosmology(paramDict=self.params, constDict=self.constdict, clTTFixFile=self.clttfile)
        clst = Clustering(self.inifile, self.expname, self.gridname, self.version, cc)
        self.ks = clst.HMF.kh
        self.ps_fid = clst.fine_ps_bar(self.mus)
        self.veff_fid = clst.V_eff(self.mus)

        self.deltaks = np.gradient(self.ks)
        self.deltamus = np.gradient(self.mus)

    def _pspec(self, params):
        cc = ClusterCosmology(paramDict=params, constDict=self.constdict, clTTFixFile=self.clttfile)
        clst = Clustering(self.inifile, self.expname, self.gridname, self.version, cc)
        return clst.fine_ps_bar(self.mus)

    def make_derivs(self):
        ps_ups = np.array(list(map(self._pspec, self.param_ups)))
        print("Done calculating ups...")
        ps_downs = np.array(list(map(self._pspec, self.param_downs)))
        print("Done calculating downs...")

        ks = self.ks[..., np.newaxis, np.newaxis]
        deltaks = self.deltaks[..., np.newaxis, np.newaxis]
        deltamus = self.deltamus[np.newaxis, np.newaxis, ...]

        print(f"ks: {ks.shape} \ndeltaks: {deltaks.shape} \ndeltamus: {deltamus.shape}")
        fisher_factors = (self.ps_fid**2 * self.veff_fid * ks**2 * deltaks * deltamus) / (8 * np.pi) 

        steps = self.steps[..., np.newaxis, np.newaxis, np.newaxis]

        derivatives = (ps_ups - ps_downs) / (2 * steps)

        return derivatives, fisher_factors

if __name__ == '__main__':
    INIFILE = "input/pipeline.ini"
    DIR = "datatest/"
    FISH_FAC_NAME = "fish_factor.npy"
    FISH_DERIV_NAME = "fish_derivs.npy" 

    deriv = Derivs_Clustering(INIFILE)
    deriv.instantiate_params()
    deriv.instantiate_grids()

    fish_derivs, fish_facs = deriv.make_derivs()

    np.save(DIR + deriv.saveid + FISH_FAC_NAME, fish_facs)
    np.save(DIR + deriv.saveid + FISH_DERIV_NAME, fish_derivs)

