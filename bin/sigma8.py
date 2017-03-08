import matplotlib
matplotlib.use('Agg')
import numpy as np
import sys, os, time
from szlib.szcounts import ClusterCosmology,SZ_Cluster_Model,Halo_MF,getTotN
from orphics.tools.io import Plotter,dictFromSection,listFromConfig
from ConfigParser import SafeConfigParser 
import cPickle as pickle
from orphics.tools.io import Plotter
from orphics.analysis.flatMaps import interpolateGrid

expName = sys.argv[1]
calName = sys.argv[2]
calFile = sys.argv[3]

#calFile = "data/S4-5mCMB_all_coarse.pkl"
#expName = "S4-5m"
#calName = "CMB_all_coarse"

# zout = np.arange(0.1,3.0,0.5)
# mout = np.arange(14.0,15.2,0.5)

iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)

fparams = {}   # the 
for (key, val) in Config.items('params'):
    if ',' in val:
        param, step = val.split(',')
        fparams[key] = float(param)
    else:
        fparams[key] = float(val)


suffix = Config.get('general','suffix')
# load the mass calibration grid
mexprange, zrange, lndM = pickle.load(open(calFile,"rb"))


saveId = expName + "_" + calName + "_" + suffix

from orphics.tools.io import dictFromSection, listFromConfig
constDict = dictFromSection(Config,'constants')
clusterDict = dictFromSection(Config,'cluster_params')
beam = listFromConfig(Config,expName,'beams')
noise = listFromConfig(Config,expName,'noises')
freq = listFromConfig(Config,expName,'freqs')
lknee = listFromConfig(Config,expName,'lknee')[0]
alpha = listFromConfig(Config,expName,'alpha')[0]

clttfile = Config.get('general','clttfile')

# get s/n q-bins
qs = listFromConfig(Config,'general','qbins')
qspacing = Config.get('general','qbins_spacing')
if qspacing=="log":
    qbins = np.logspace(np.log10(qs[0]),np.log10(qs[1]),int(qs[2]))
elif qspacing=="linear":
    qbins = np.linspace(qs[0],qs[1],int(qs[2]))
else:
    raise ValueError



cc = ClusterCosmology(fparams,constDict,clTTFixFile=clttfile)
HMF = Halo_MF(cc,mexprange,zrange)
SZProf = SZ_Cluster_Model(cc,clusterDict,rms_noises = noise,fwhms=beam,freqs=freq,lknee=lknee,alpha=alpha)
dNFid_dmqz = HMF.N_of_mqz_SZ(lndM,qbins,SZProf)

origPk = HMF.pk.copy()

h = 0.1

derivs = []

for i,z in enumerate(zrange):
    print "Calculating derivatives for redshift ", z
    HMF.pk = origPk.copy()
    HMF.pk[i,:] *= (1.+h/2.)
    dNUp_dmqz = HMF.N_of_mqz_SZ(lndM,qbins,SZProf)


    HMF.pk = origPk.copy()
    HMF.pk[i,:] *= (1.-h/2.)
    dNDn_dmqz = HMF.N_of_mqz_SZ(lndM,qbins,SZProf)


    derivN_mqz = (dNUp_dmqz-dNDn_dmqz)/h


    derivs.append(derivN_mqz)


import itertools
zindices = range(zrange.size)
lenz = zrange.size
# paramCombs = itertools.combinations_with_replacement(zindices,2)
fsky = 0.4
N_fid = dNFid_dmqz*fsky



# Fisher params
fishSection = 'fisher-lcdm'
zlist = ["S8Z"+str(z) for z in zrange]
paramList = Config.get(fishSection,'paramList').split(',')+zlist
saveName = Config.get(fishSection,'saveSuffix')
numParams = len(paramList)
Fisher = np.zeros((numParams,numParams))
paramCombs = itertools.combinations_with_replacement(paramList,2)


# Populate Fisher
for param1,param2 in paramCombs:
    if param1=='tau' or param2=='tau': continue


    if param1[:3]=="S8Z":
        k1 = zlist.index(param1)
        dN1 = derivs[k1][:,:,:]*fsky
    else:
        dN1 = np.load("data/dN_dzmq_"+saveId+"_"+param1+".npy")
        dN1 = dN1[:,:,:]*fsky

    if param2[:3]=="S8Z":
        k2 = zlist.index(param1)
        dN2 = derivs[k2][:,:,:]*fsky
    else:
        dN2 = np.load("data/dN_dzmq_"+saveId+"_"+param2+".npy")
        dN2 = dN2[:,:,:]*fsky


    i = paramList.index(param1)
    j = paramList.index(param2)

    assert not(np.any(np.isnan(dN1)))
    assert not(np.any(np.isnan(dN2)))
    assert not(np.any(np.isnan(N_fid)))


    with np.errstate(divide='ignore'):
        FellBlock = dN1*dN2*np.nan_to_num(1./N_fid)
    Fellnoq = np.trapz(FellBlock,qbins,axis=2)
    Fellnoz = np.trapz(Fellnoq,zrange,axis=1)
    Fell = np.trapz(Fellnoz,10**mexprange)

    
       
    Fisher[i,j] = Fell
    Fisher[j,i] = Fell    


print Fisher.shape

np.savetxt("output/fisherSigma8"+saveId+".txt",Fisher)

Aerrs = np.sqrt(np.diagonal(np.linalg.inv(Fisher)))

pl = Plotter()
pl.addErr(zrange,zrange*0.+1.,yerr=Aerrs)
pl.done("output/aerrs"+saveId+".png")

