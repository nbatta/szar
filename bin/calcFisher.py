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

ms = listFromConfig(Config,calName,'mexprange')
mgrid = np.arange(ms[0],ms[1],ms[2])
zs = listFromConfig(Config,calName,'zrange')
zgrid = np.arange(zs[0],zs[1],zs[2])

dm = np.diff(10**mgrid)
dz = np.diff(zgrid)


paramList = [] # the parameters that can be varied
fparams = {}   # the 
stepSizes = {}
for (key, val) in Config.items('params'):
    if ',' in val:
        param, step = val.split(',')
        paramList.append(key)
        fparams[key] = float(param)
        stepSizes[key] = float(step)
    else:
        fparams[key] = float(val)






paramList = Config.get('fisher','paramList').split(',')
numParams = len(paramList)
Fisher = np.zeros((numParams,numParams))
paramCombs = itertools.combinations_with_replacement(paramList,2)

N_fid = np.load("data/N_dzmq_"+saveId+"_fid"+".npy")
print getTotN(N_fid,mgrid,zgrid,qbins)
print N_fid.shape
#N_fid = debin(N_fid,mgrid,zgrid,qbins)

fisherPlanck = np.loadtxt("data/Feb18_FisherMat_Planck_tau0.01_lens_fsky0.6.csv")

for param1,param2 in paramCombs:
    if param1=='tau' or param2=='tau': continue
    dN1 = np.load("data/dN_dzmq_"+saveId+"_"+param1+".npy")
    dN2 = np.load("data/dN_dzmq_"+saveId+"_"+param2+".npy")


    # N1up = np.load("data/dNup_dzmq_"+saveId+"_"+param1+".npy")
    # N1dn = np.load("data/dNdn_dzmq_"+saveId+"_"+param1+".npy")
    # N2up = np.load("data/dNup_dzmq_"+saveId+"_"+param2+".npy")
    # N2dn = np.load("data/dNdn_dzmq_"+saveId+"_"+param2+".npy")

    # N1up = debin(N1up,mgrid,zgrid,qbins)
    # N2up = debin(N2up,mgrid,zgrid,qbins)
    # N1dn = debin(N1dn,mgrid,zgrid,qbins)
    # N2dn = debin(N2dn,mgrid,zgrid,qbins)

    # dN1 = (N1up-N1dn)/stepSizes[param1]
    # dN2 = (N2up-N2dn)/stepSizes[param2]

    i = paramList.index(param1)
    j = paramList.index(param2)


    # if i==j:
    #     print param1, getTotN(N1up,mgrid,zgrid,qbins)
    #     print param1, getTotN(N1dn,mgrid,zgrid,qbins)

    assert not(np.any(np.isnan(dN1)))
    assert not(np.any(np.isnan(dN2)))
    assert not(np.any(np.isnan(N_fid)))

    #Fell = np.nansum(dN1*dN2/N_fid)
    #Fell = np.sum(dN1*dN2*np.nan_to_num(1./N_fid))

    FellBlock = dN1*dN2*np.nan_to_num(1./N_fid)
    Fellnoq = np.trapz(FellBlock,qbins,axis=2)
    Fellnoz = np.trapz(Fellnoq,zgrid,axis=1)
    Fell = np.trapz(Fellnoz,10**mgrid)

    # Fell = getTotN(dN1*dN2*np.nan_to_num(1./N_fid),mgrid,zgrid,qbins)

    # Fellnoq = FellBlock.sum(axis=2)
    # Fellnoz = Fellnoq.sum(axis=1)
    # Fell = Fellnoz.sum()
    
       
    Fisher[i,j] = Fell
    Fisher[j,i] = Fell    

fisherPlanck = np.pad(fisherPlanck,pad_width=((0,5),(0,5)),mode="constant",constant_values=0.)
#print fisherPlanck
Fisher += fisherPlanck

#print Fisher.shape
#print Fisher
Finv = np.linalg.inv(Fisher)

# pl = Plotter()
# pl.plot2d(Finv)
# pl.done("output/fisher.png")

errs = np.sqrt(np.diagonal(Finv))
errDict = {}
for i,param in enumerate(paramList):
    errDict[param] = errs[i]


print "Mnu 1-sigma : ", errDict['mnu']*1000 , " meV"

