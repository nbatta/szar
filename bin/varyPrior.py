import matplotlib
matplotlib.use('Agg')
from configparser import SafeConfigParser 
import pickle as pickle
import numpy as np
import sys
from orphics.tools.io import dictFromSection, listFromConfig
from orphics.tools.io import Plotter
import matplotlib.pyplot as plt
from szar.fisher import getFisher
from szar.counts import rebinN


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
saveId = expName + "_" + gridName + "_" + calName + "_v" + version



# get s/n q-bins
qs = listFromConfig(Config,'general','qbins')
qspacing = Config.get('general','qbins_spacing')
if qspacing=="log":
    qbins = np.logspace(np.log10(qs[0]),np.log10(qs[1]),int(qs[2]))
elif qspacing=="linear":
    qbins = np.linspace(qs[0],qs[1],int(qs[2]))
else:
    raise ValueError
dq = np.diff(qbins)

fsky = Config.getfloat(expName,'fsky')

# get mass and z grids
ms = listFromConfig(Config,gridName,'mexprange')
mexp_edges = np.arange(ms[0],ms[1]+ms[2],ms[2])
zs = listFromConfig(Config,gridName,'zrange')
z_edges = np.arange(zs[0],zs[1]+zs[2],zs[2])
dm = np.diff(10**mexp_edges)
dz = np.diff(z_edges)

# Fisher params
fishSection = 'fisher-'+fishName
paramList = Config.get(fishSection,'paramList').split(',')
paramLatexList = Config.get(fishSection,'paramLatexList').split(',')
saveName = Config.get(fishSection,'saveSuffix')

# Fiducial number counts
new_z_edges, N_fid = rebinN(np.load(bigDataDir+"N_mzq_"+saveId+"_fid"+".npy"),pzcutoff,z_edges)

N_fid = N_fid[:,:,:]*fsky
print(("Total number of clusters: ", N_fid.sum())) #getTotN(N_fid,mgrid,zgrid,qbins)


sId = expName + "_" + gridName  + "_v" + version
#sovernsquareEach = np.loadtxt(bigDataDir+"sampleVarGrid_"+sId+".txt")
#sovernsquare =  np.dstack([sovernsquareEach]*len(qbins))


# Planck and BAO Fishers
planckFile = Config.get(fishSection,'planckFile')
try:
    baoFile = Config.get(fishSection,'baoFile')
except:
    baoFile = ''

# Number of non-SZ params (params that will be in Planck/BAO)
numCosmo = Config.getint(fishSection,'numCosmo')
numLeft = len(paramList) - numCosmo
fisherPlanck = 0.
if planckFile!='':
    try:
        fisherPlanck = np.loadtxt(planckFile)
    except:
        fisherPlanck = np.loadtxt(planckFile,delimiter=',')
    fisherPlanck = np.pad(fisherPlanck,pad_width=((0,numLeft),(0,numLeft)),mode="constant",constant_values=0.)



from collections import OrderedDict
priorList = OrderedDict()
priorList['tau'] = 0.1
priorList['H0'] = 10.0
priorList['omch2'] = 0.002
priorList['ombh2'] = 0.00023
priorList['ns'] = 0.006
priorList['As'] = 5.e-12

priorList['alpha_ym'] = 0.179
priorList['b_wl'] = 0.1
priorList['b_ym'] = 0.08
priorList['beta_ym'] = 0.1
priorList['gamma_ym'] = 0.1
priorList['Ysig'] = 0.0127
priorList['gammaYsig'] = 0.1
priorList['betaYsig'] = 1.0


import os
if fishName=='mnu':            
    pl = Plotter(labelY="$\sigma("+paramLatexList[paramList.index(fishName)]+")$",labelX="Iteration",ftsize=12)
elif fishName=='w0':
    pl = Plotter(labelY="$\\frac{\sigma("+paramLatexList[paramList.index(fishName)]+")}{"+paramLatexList[paramList.index(fishName)]+"}\%$",labelX="Iteration",ftsize=12)

priorNameList = []
priorValueList = []
iterations = 0

numlogs = 30
pertol = 0.1
mink = 5
perRange = np.logspace(-4,2,numlogs)[::-1]



for prior in list(priorList.keys()):
    priorNameList.append(prior)

    preVal = np.inf
    priorRange = perRange*priorList[prior]/100.
    priorValueList.append(priorRange[0])
    print((priorNameList, priorValueList))
    sigs = []
    xs = []
    k = 0
    for val in priorRange:
        iterations += 1
        xs.append(iterations)
        FisherTot = 0.
        priorValueList[-1] = val

        ##########################
        # Populate Fisher
        Fisher = getFisher(N_fid,paramList,priorNameList,priorValueList,bigDataDir,saveId,pzcutoff,z_edges,fsky)
        ##########################


        fisherBAO = Fisher.copy()*0.
        if baoFile!='':
            try:
                fisherBAO = np.loadtxt(baoFile)
            except:
                fisherBAO = np.loadtxt(baoFile,delimiter=',')
            fisherBAO = np.pad(fisherBAO,pad_width=((0,numLeft),(0,numLeft)),mode="constant",constant_values=0.)


        FisherTot = Fisher + fisherPlanck
        FisherTot += fisherBAO


        Finv = np.linalg.inv(FisherTot)

        errs = np.sqrt(np.diagonal(Finv))
        errDict = {}
        for i,param in enumerate(paramList):
            errDict[param] = errs[i]

        if fishName=='mnu':            
            constraint = errDict[fishName]*1000
        elif fishName=='w0':
            constraint = errDict[fishName]*100
        sigs.append(constraint)
        if (np.abs(preVal-constraint)*100./constraint)<pertol:
            print(((constraint-preVal)*100./constraint))
            if k>mink: break
        preVal = constraint
        print((prior, val,constraint))
        k+=1

    priorLabel = paramLatexList[paramList.index(prior)]
    pl.add(xs,sigs,label="$"+priorLabel+"$")
pl.legendOn(loc='upper right',labsize=8)
pl.done(os.environ['WWW']+"plots/priors_"+fishName+".pdf")
