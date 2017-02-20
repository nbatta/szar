import matplotlib
matplotlib.use('Agg')
import itertools
from ConfigParser import SafeConfigParser 
import cPickle as pickle
import numpy as np
import sys
from orphics.tools.io import dictFromSection, listFromConfig
from orphics.tools.io import Plotter

def getTotN(Nmqz,mgrid,zgrid,qbins):

    Fellnoq = np.trapz(Nmqz,qbins,axis=2)
    Fellnom = np.trapz(Fellnoq.T,10**mgrid,axis=1)
    # pl = Plotter()
    # pl.add(zgrid,Fellnom)
    # pl.done("output/nz.png")
    N = np.trapz(Fellnom.T,zgrid)
    return N


def debin(Nmqz,mgrid,zgrid,qbins):

    dm = np.diff(10**mgrid).reshape((Nmqz.shape[0]-1,1,1))
    dq = np.diff(qbins).reshape((1,1,Nmqz.shape[2]-1))
    dz = np.diff(zgrid).reshape((1,Nmqz.shape[1]-1,1))
    N = Nmqz[:-1,:-1,:-1] * dm *dq *dz
    

    return N



expName = sys.argv[1]
calName = sys.argv[2]

iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)

suffix = Config.get('general','suffix')
saveId = expName + "_" + calName + "_" + suffix



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

zmax = Config.getfloat('general','zmax')
fsky = Config.getfloat(expName,'fsky')
ms = listFromConfig(Config,calName,'mexprange')
mgrid = np.arange(ms[0],ms[1],ms[2])
zs = listFromConfig(Config,calName,'zrange')
zgrid = np.arange(zs[0],zs[1],zs[2])

zgrid = zgrid[zgrid<zmax]
zlen = zgrid.size

dm = np.diff(10**mgrid)
dz = np.diff(zgrid)


paramList = Config.get('fisher','paramList').split(',')
numParams = len(paramList)
Fisher = np.zeros((numParams,numParams))
paramCombs = itertools.combinations_with_replacement(paramList,2)

N_fid = np.load("data/N_dzmq_"+saveId+"_fid"+".npy")
N_fid = N_fid[:,:zlen,:]*fsky
print "Total number of clusters: ", getTotN(N_fid,mgrid,zgrid,qbins)

planckFile = Config.get('fisher','planckFile')
baoFile = Config.get('fisher','baoFile')

numCosmo = Config.getint('fisher','numCosmo')
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


for param1,param2 in paramCombs:
    if param1=='tau' or param2=='tau': continue
    dN1 = np.load("data/dN_dzmq_"+saveId+"_"+param1+".npy")
    dN2 = np.load("data/dN_dzmq_"+saveId+"_"+param2+".npy")
    dN1 = dN1[:,:zlen,:]*fsky
    dN2 = dN2[:,:zlen,:]*fsky

    i = paramList.index(param1)
    j = paramList.index(param2)


    assert not(np.any(np.isnan(dN1)))
    assert not(np.any(np.isnan(dN2)))
    assert not(np.any(np.isnan(N_fid)))

    # if param1=='w0':
    #     Nup = np.load("data/dNup_dzmq_"+saveId+"_"+param1+".npy")
    #     Ndn = np.load("data/dNdn_dzmq_"+saveId+"_"+param1+".npy")
    #     print "Total number of clusters: ", getTotN(Nup,mgrid,zgrid,qbins)
    #     print "Total number of clusters: ", getTotN(Ndn,mgrid,zgrid,qbins)

    with np.errstate(divide='ignore'):
        FellBlock = dN1*dN2*np.nan_to_num(1./N_fid)
    Fellnoq = np.trapz(FellBlock,qbins,axis=2)
    Fellnoz = np.trapz(Fellnoq,zgrid,axis=1)
    Fell = np.trapz(Fellnoz,10**mgrid)

    
       
    Fisher[i,j] = Fell
    Fisher[j,i] = Fell    

Fisher += fisherPlanck
Fisher += fisherBAO

Finv = np.linalg.inv(Fisher)


errs = np.sqrt(np.diagonal(Finv))
errDict = {}
for i,param in enumerate(paramList):
    errDict[param] = errs[i]


try:
    print "Mnu 1-sigma : "+ str(errDict['mnu']*1000) + " meV"
except:
    pass
try:
    print "w 1-sigma : "+ str(errDict['w0']*100.) +" %"
except:
    pass

