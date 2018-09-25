from __future__ import print_function
from __future__ import division
from future import standard_library
standard_library.install_aliases()
from past.utils import old_div
import numpy as np
from szar.counts import ClusterCosmology,Halo_MF
from szar.szproperties import SZ_Cluster_Model
from szar.clustering import Clustering
from configparser import SafeConfigParser
from orphics.io import dict_from_section,list_from_config

try:
    import cPickle as pickle
except ImportError:
    import pickle

import matplotlib.pyplot as plt

# Load ini
iniFile = 'input/pipeline.ini'
expName = 'S4-1.0-CDT' 
gridName = 'grid-owl2' 
version = '0.6' 
clst = Clustering(iniFile,expName,gridName,version)

M200temp = np.arange(1,1000,1) * 1e11

print(clst.non_linear_scale(1.95,M200temp))

print(clst.ntilde())
print(clst.b_eff_z())
print(clst.Norm_Sfunc(0.4))

zarrs = np.arange(0,4,0.05)
fg = clst.cc.fgrowth(zarrs) #?Is this a calculated growth (as in numerically)?

aarrs = old_div(1.,(1. + zarrs))

g = lambda x: (0.3*(1 + x)**3)**0.55 #?Is this the analytic growth function?



thk = 3 #Controls plot linewidth

plt.figure(figsize=(10,8))
plt.rc('axes', linewidth=thk)
plt.tick_params(which='both',size=14,width=thk,labelsize = 24)
plt.tick_params(which='major',size=18,width=thk,labelsize = 24)
plt.xlabel(r'$z$', fontsize=32,weight='bold')
plt.ylabel(r'$f_g$', fontsize=32,weight='bold')
plt.plot(zarrs,g(zarrs),linewidth=thk) #?Compare analytic growth against...?
plt.plot(zarrs,fg,linewidth=thk) #?...the calculated?
plt.savefig('testing_fg.png', bbox_inches='tight',format='png')
