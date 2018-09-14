from __future__ import print_function
import numpy as np
from szar.counts import ClusterCosmology,Halo_MF
from szar.szproperties import SZ_Cluster_Model
from szar.clustering import clustering
from configparser import SafeConfigParser
from orphics.io import dict_from_section,list_from_config
import cPickle as pickle
import matplotlib.pyplot as plt


iniFile = 'input/pipeline.ini'
expName = 'S4-1.0-CDT'
gridName = 'grid-owl2'
version = '0.7'
clst = clustering(iniFile,expName,gridName,version)

M200temp = np.arange(1,1000,1) * 1e11

#print M200temp
print(clst.non_linear_scale(1.95,M200temp))

#print clst.HMF.M200.shape
print(clst.ntilde())
print(clst.b_eff_z())
print(clst.Norm_Sfunc(0.4))

zarrs = np.arange(0,4,0.05)
#print clst.cc.GrowthFunc(zarrs)
fg = clst.cc.fgrowth(zarrs)

aarrs = 1./(1. + zarrs)

g = lambda x: (0.3*(1 + x)**3)**0.55



thk = 3

plt.figure(figsize=(10,8))
plt.rc('axes', linewidth=thk)
plt.tick_params(which='both',size=14,width=thk,labelsize = 24)
plt.tick_params(which='major',size=18,width=thk,labelsize = 24)
plt.xlabel(r'$z$', fontsize=32,weight='bold')
plt.ylabel(r'$f_g$', fontsize=32,weight='bold')
plt.plot(zarrs,g(zarrs),linewidth=thk)
plt.plot(zarrs,fg,linewidth=thk)
plt.savefig('testing_fg.png', bbox_inches='tight',format='png')
