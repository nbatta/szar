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
fishName = sys.argv[4]

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

# Fisher params
fishSection = 'fisher-'+fishName
saveName = Config.get(fishSection,'saveSuffix')

#with nostdout():
actualN = sfisher.counts_from_config(Config,bigDataDir,version,expName,gridName,mexp_edges,z_edges)

print("Actual number of clusters: ", actualN)


##########################
# Populate Fisher
FisherTot, paramList = sfisher.cluster_fisher_from_config(Config,expName,gridName,calName,fishName)
##########################

print(FisherTot[6:8,6:8], paramList[6:8])

if (YWLcorrflag == 0 and FisherTot[-1,-1] == 0):
    FisherTot = FisherTot[:len(paramList)-2,:len(paramList)-2]
    paramList = paramList[:len(paramList)-2]

pickle.dump((paramList,FisherTot),open(bigDataDir+"savedFisher_"+saveId+"_"+saveName+".pkl",'wb'))
    
Finv = np.linalg.inv(FisherTot)

print(np.linalg.det(Finv[6:8,6:8]))
print("FOM", old_div(1.,np.sqrt(np.linalg.det(Finv[6:8,6:8]))))

errs = np.sqrt(np.diagonal(Finv))
errDict = {}
for i,param in enumerate(paramList):
    errDict[param] = errs[i]


try:
    print("(1-b) 1-sigma : "+ str(errDict['b_ym']*100./0.8) + " %")
except:
    pass


try:
    print("Mnu 1-sigma : "+ str(errDict['mnu']*1000) + " meV")
except:
    pass
try:
    print("w0 1-sigma : "+ str(errDict['w0']*100.) +" %")
except:
    pass
try:
    print("wa 1-sigma : "+ str(errDict['wa'])) 
except:
    pass

try:
    print("bMWL 1-sigma : "+ str(errDict['b_wl']*100.)  + " %")
except:
    pass

try:
    print("sigma8 1-sigma : "+ str(errDict['S8All']))
except:
    pass

try:
    print("sigR 1-sigma : "+ str(errDict['sigR']))
except:
    pass




