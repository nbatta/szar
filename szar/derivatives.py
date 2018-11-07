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
from orphics.io import dict_from_section, list_from_config
import numpy as np

# etc
import time
import sys

class Derivs_Clustering(object):
    def __init__(self, inifile, expname, gridname, version):
        self.params = None
        self.inifile = inifile
        self.config = ConfigParser()
        self.config.optionxform=str
        self.config.read(self.inifile)

        self.constdict = dict_from_section(self.config,'constants')
        self.clttfile = self.config.get('general','clttfile')

        self.expname = expname
        self.gridname = gridname
        self.version = version

        self.saveid = self.expname + "_" + self.gridname + "_v" + self.version

    def instantiate_params(self, selected_params=None):
        manual_param_list = self.config.get('general','manualParams').split(',')

        paramdict = {}
        stepsizes = {}
        for (key, val) in self.config.items('params'):
            if key in manual_param_list:
                continue

            if ',' in val:
                param, step = val.split(',')
                paramdict[key] = float(param)
                stepsizes[key] = float(step)/2
            else:
                paramdict[key] = float(val)
                stepsizes[key] = None

        if selected_params is None:
            selected_params = paramdict
        else:
            selected_params = {key:paramdict[key] for key in selected_params}
            for key in selected_params.keys():
                if stepsizes[key] is None:
                    print("You selected a param that has no defined stepsize!")
                    sys.exit()
        
        param_ups = []
        param_downs = []
        for key,val in selected_params.items():
            if stepsizes[key] is not None:
                param_up = paramdict.copy()
                param_down = paramdict.copy()

                param_up[key] = val + stepsizes[key]
                param_down[key] = val - stepsizes[key]

                param_ups.append(param_up)
                param_downs.append(param_down)

        self.params = paramdict
        self.param_ups = param_ups
        self.param_downs = param_downs
        self.fisher_params = {key:value for key,value in selected_params.items() if stepsizes[key] is not None}
        stepdict = {key:value for key,value in stepsizes.items() if value is not None}
        self.steps = np.fromiter(stepdict.values(), dtype=float)

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

        fisher_factors = (self.veff_fid * ks**2 * deltaks * deltamus) / (8 * np.pi**2 * self.ps_fid**2) 

        steps = self.steps[..., np.newaxis, np.newaxis, np.newaxis]

        derivatives = (ps_ups - ps_downs) / (2 * steps)

        return derivatives, fisher_factors, ps_ups, ps_downs

def make_fisher(derivs, prefactors):
    fisher_terms = prefactors[np.newaxis, ...] * derivs
    fisher_mat = np.einsum('aijk,bijk->ab', derivs, fisher_terms)
    return fisher_mat
