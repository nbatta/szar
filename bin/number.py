from __future__ import print_function
from __future__ import division
from builtins import str
from past.utils import old_div
import matplotlib
matplotlib.use('Agg')
import traceback
from configparser import SafeConfigParser 
import pickle as pickle
import numpy as np
import sys
from orphics.io import dict_from_section, list_from_config
from orphics.io import Plotter, nostdout
import matplotlib.pyplot as plt
from szar.fisher import getFisher
from szar.counts import rebinN
import szar.fisher as sfisher
from szar.counts import ClusterCosmology,Halo_MF
from szar.szproperties import SZ_Cluster_Model


expName = sys.argv[1]
gridName = sys.argv[2]
calName = sys.argv[3]

iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)
bigDataDir = Config.get('general','bigDataDirectory')

version = Config.get('general','version')
pzcutoff = Config.getfloat('general','photoZCutOff')
saveId = sfisher.save_id(expName,gridName,calName,version)
derivRoot = sfisher.deriv_root(bigDataDir,saveId)
YWLcorrflag = Config.getfloat('general','ywl_corr_flag')    

# get mass and z grids
ms = list_from_config(Config,gridName,'mexprange')
mexp_edges = np.arange(ms[0],ms[1]+ms[2],ms[2])
zs = list_from_config(Config,gridName,'zrange')
z_edges = np.arange(zs[0],zs[1]+zs[2],zs[2])


zmin = -np.inf #0.9
zmax = np.inf #1.1
mmin = -np.inf #13.9
mmax = np.inf #14.1

#with nostdout():
actualN = sfisher.sel_counts_from_config(Config,bigDataDir,version,expName,gridName,calName,mexp_edges,z_edges,zmin=zmin,zmax=zmax,mmin=mmin,mmax=mmax).sum()


print("Actual number of clusters: ", actualN)

