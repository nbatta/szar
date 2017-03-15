import matplotlib
matplotlib.use('Agg')
import itertools
from ConfigParser import SafeConfigParser 
import cPickle as pickle
import numpy as np
import sys
from orphics.tools.io import dictFromSection, listFromConfig
from orphics.tools.io import Plotter
import matplotlib.pyplot as plt

from szlib.szcounts import rebinN




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
saveName = Config.get(fishSection,'saveSuffix')
numParams = len(paramList)
Fisher = np.zeros((numParams,numParams))
paramCombs = itertools.combinations_with_replacement(paramList,2)

# Fiducial number counts
new_z_edges, N_fid = rebinN(np.load(bigDataDir+"N_mzq_"+saveId+"_fid"+".npy"),pzcutoff,z_edges)

N_fid = N_fid[:,:,:]*fsky
print "Total number of clusters: ", N_fid.sum() #getTotN(N_fid,mgrid,zgrid,qbins)


sId = expName + "_" + gridName  + "_v" + version
sovernsquareEach = np.loadtxt(bigDataDir+"sampleVarGrid_"+sId+".txt")
sovernsquare =  np.dstack([sovernsquareEach]*len(qbins))


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

fisherBAO = 0.
if baoFile!='':
    try:
        fisherBAO = np.loadtxt(baoFile)
    except:
        fisherBAO = np.loadtxt(baoFile,delimiter=',')
    fisherBAO = np.pad(fisherBAO,pad_width=((0,numLeft),(0,numLeft)),mode="constant",constant_values=0.)


    

    
# Populate Fisher
for param1,param2 in paramCombs:
    if param1=='tau' or param2=='tau': continue
    new_z_edges, dN1 = rebinN(np.load(bigDataDir+"dNdp_mzq_"+saveId+"_"+param1+".npy"),pzcutoff,z_edges)
    new_z_edges, dN2 = rebinN(np.load(bigDataDir+"dNdp_mzq_"+saveId+"_"+param2+".npy"),pzcutoff,z_edges)
    dN1 = dN1[:,:,:]*fsky
    dN2 = dN2[:,:,:]*fsky


    i = paramList.index(param1)
    j = paramList.index(param2)

    if param1=='wa':
        Nup = np.load(bigDataDir+"Nup_mzq_"+saveId+"_"+param1+".npy")
        Ndn = np.load(bigDataDir+"Ndn_mzq_"+saveId+"_"+param1+".npy")
        print Nup.sum()
        print Ndn.sum()

    assert not(np.any(np.isnan(dN1)))
    assert not(np.any(np.isnan(dN2)))
    assert not(np.any(np.isnan(N_fid)))


    with np.errstate(divide='ignore'):
        FellBlock = dN1*dN2*np.nan_to_num(1./(N_fid))#+(N_fid*N_fid*sovernsquare)))
    Fell = FellBlock.sum()

    
       
    Fisher[i,j] = Fell
    Fisher[j,i] = Fell    


FisherTot = Fisher + fisherPlanck
FisherTot += fisherBAO

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




pickle.dump((paramList,FisherTot),open(bigDataDir+"savedFisher_"+saveId+"_"+saveName+".pkl",'wb'))
