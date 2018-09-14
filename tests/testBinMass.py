from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import traceback
from configparser import SafeConfigParser 
import pickle as pickle
import numpy as np
import sys
from orphics.tools.io import dictFromSection, listFromConfig
from orphics.tools.io import Plotter, nostdout
import matplotlib.pyplot as plt
from szar.fisher import getFisher
from szar.counts import rebinN
import szar.fisher as sfisher
from szar.counts import ClusterCosmology,Halo_MF
from szar.szproperties import SZ_Cluster_Model


iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)
bigDataDir = Config.get('general','bigDataDirectory')

version = Config.get('general','version')
pzcutoff = Config.getfloat('general','photoZCutOff')

gridName = "grid-default"

# get mass and z grids
ms = listFromConfig(Config,gridName,'mexprange')
mexp_edges = np.arange(ms[0],ms[1]+ms[2],ms[2])
zs = listFromConfig(Config,gridName,'zrange')
z_edges = np.arange(zs[0],zs[1]+zs[2],zs[2])


M_edges = 10**mexp_edges
Masses = (M_edges[1:]+M_edges[:-1])/2.

dm = np.diff(M_edges)
print(Masses)
print((dm*100./Masses))
