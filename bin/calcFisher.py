import matplotlib
matplotlib.use('Agg')
import traceback
from ConfigParser import SafeConfigParser 
import cPickle as pickle
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
    


fsky = Config.getfloat(expName,'fsky')

# get mass and z grids
ms = listFromConfig(Config,gridName,'mexprange')
mexp_edges = np.arange(ms[0],ms[1]+ms[2],ms[2])
zs = listFromConfig(Config,gridName,'zrange')
z_edges = np.arange(zs[0],zs[1]+zs[2],zs[2])

# Fisher params
fishSection = 'fisher-'+fishName
saveName = Config.get(fishSection,'saveSuffix')

with nostdout():
    actualN = sfisher.counts_from_config(Config,bigDataDir,version,expName,gridName,mexp_edges,z_edges)

print "Actual number of clusters: ", actualN


# Planck and BAO Fishers
planckFile = Config.get(fishSection,'planckFile')
try:
    baoFile = Config.get(fishSection,'baoFile')
except:
    baoFile = ''

    
    
# Number of non-SZ params (params that will be in Planck/BAO)
numCosmo = Config.getint(fishSection,'numCosmo')


##########################
# Populate Fisher
Fisher, paramList = sfisher.cluster_fisher_from_config(Config,expName,gridName,calName,fishName)
##########################



numLeft = len(paramList) - numCosmo



fisherPlanck = 0.
if planckFile!='':
    try:
        fisherPlanck = np.loadtxt(planckFile)
    except:
        fisherPlanck = np.loadtxt(planckFile,delimiter=',')
    fisherPlanck = sfisher.pad_fisher(fisherPlanck,numLeft)





fisherBAO = Fisher.copy()*0.
if baoFile!='':
    try:
        fisherBAO = np.loadtxt(baoFile)
    except:
        fisherBAO = np.loadtxt(baoFile,delimiter=',')
    fisherBAO = sfisher.pad_fisher(fisherBAO,numLeft)


FisherTot = Fisher + fisherPlanck
FisherTot += fisherBAO

try:
    otherFishers = Config.get(fishSection,'otherFishers').split(',')
    for otherFisherFile in otherFishers:
        try:
            other_fisher = np.loadtxt(otherFisherFile)
        except:
            other_fisher = np.loadtxt(otherFisherFile,delimiter=',')
        other_fisher = sfisher.pad_fisher(other_fisher,numLeft)
        FisherTot += other_fisher
            
        
except:
    traceback.print_exc()
    print "No other fishers found."

    

# from orphics.tools.io import Plotter
# import os
# pl = Plotter()
# pl.plot2d(np.log10(np.abs(fisherPlanck)))
# pl.done(os.environ['WWW']+"fisher.png")          

Finv = np.linalg.inv(FisherTot)

errs = np.sqrt(np.diagonal(Finv))
errDict = {}
for i,param in enumerate(paramList):
    errDict[param] = errs[i]


try:
    print "(1-b) 1-sigma : "+ str(errDict['b_ym']*100./0.8) + " %"
except:
    pass


try:
    print "Mnu 1-sigma : "+ str(errDict['mnu']*1000) + " meV"
except:
    pass
try:
    print "w0 1-sigma : "+ str(errDict['w0']*100.) +" %"
except:
    pass
try:
    print "wa 1-sigma : "+ str(errDict['wa']) 
except:
    pass

try:
    print "bMWL 1-sigma : "+ str(errDict['b_wl']*100.)  + " %"
except:
    pass

try:
    print "sigma8 1-sigma : "+ str(errDict['S8All'])
except:
    pass

try:
    print "sigR 1-sigma : "+ str(errDict['sigR'])
except:
    pass




pickle.dump((paramList,FisherTot),open(bigDataDir+"savedFisher_"+saveId+"_"+saveName+".pkl",'wb'))
