import matplotlib
matplotlib.use('Agg')
from ConfigParser import SafeConfigParser 
import cPickle as pickle
import numpy as np
import sys
from orphics.tools.io import dictFromSection, listFromConfig
from orphics.tools.io import Plotter
import matplotlib.pyplot as plt
from szar.fisher import getFisher
from szar.counts import rebinN

from szar.counts import ClusterCosmology,Halo_MF
from szar.szproperties import SZ_Cluster_Model


def getExpN(Config,bigDataDir,version,expName,gridName,mexp_edges,z_edges):
    experimentName = expName
    cosmoDict = dictFromSection(Config,"params")
    constDict = dictFromSection(Config,'constants')
    clusterDict = dictFromSection(Config,'cluster_params')
    clttfile = Config.get("general","clttfile")
    cc = ClusterCosmology(cosmoDict,constDict,clTTFixFile = clttfile)

    beam = listFromConfig(Config,experimentName,'beams')
    noise = listFromConfig(Config,experimentName,'noises')
    freq = listFromConfig(Config,experimentName,'freqs')
    lmax = int(Config.getfloat(experimentName,'lmax'))
    lknee = float(Config.get(experimentName,'lknee').split(',')[0])
    alpha = float(Config.get(experimentName,'alpha').split(',')[0])
    fsky = Config.getfloat(experimentName,'fsky')
    SZProf = SZ_Cluster_Model(cc,clusterDict,rms_noises = noise,fwhms=beam,freqs=freq,lknee=lknee,alpha=alpha)
    hmf = Halo_MF(cc,mexp_edges,z_edges)
    mgrid,zgrid,siggrid = pickle.load(open(bigDataDir+"szgrid_"+expName+"_"+gridName+ "_v" + version+".pkl",'rb'))

    hmf.sigN = siggrid.copy()
    Ns = np.multiply(hmf.N_of_z_SZ(SZProf)*fsky,np.diff(z_edges).reshape(1,z_edges.size-1)).ravel()
    #Ns = np.multiply(hmf.N_of_z()*fsky,np.diff(z_edges).reshape(1,z_edges.size-1)).ravel()

    return Ns.sum()

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

# Fiducial number counts
new_z_edges, N_fid = rebinN(np.load(bigDataDir+"N_mzq_"+saveId+"_fid"+".npy"),pzcutoff,z_edges)

N_fid = N_fid[:,:,:]*fsky
print "Effective number of clusters: ", N_fid.sum() #getTotN(N_fid,mgrid,zgrid,qbins)
print "Actual number of clusters: ", getExpN(Config,bigDataDir,version,expName,gridName,mexp_edges,z_edges)

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





try:
    priorNameList = Config.get(fishSection,'prior_names').split(',')
    priorValueList = listFromConfig(Config,fishSection,'prior_values')
except:
    priorNameList = []
    priorValueList = []
    
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




pickle.dump((paramList,FisherTot),open(bigDataDir+"savedFisher_"+saveId+"_"+saveName+".pkl",'wb'))
