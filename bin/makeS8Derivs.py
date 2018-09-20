from __future__ import print_function
from __future__ import division
from builtins import str
from past.utils import old_div
from mpi4py import MPI
import matplotlib
matplotlib.use('Agg')
import numpy as np
import sys, os, time
from szar.counts import ClusterCosmology,Halo_MF,getNmzq,getA
from szar.szproperties import SZ_Cluster_Model
from orphics.io import Plotter,dict_from_section,list_from_config
from configparser import SafeConfigParser 
import pickle as pickle
from orphics.maps import interpolateGrid

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()    


expName = sys.argv[1]
gridName = sys.argv[2]
calName = sys.argv[3]
calFile = sys.argv[4]


iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)
version = Config.get('general','version')

fparams = {}   # the 
for (key, val) in Config.items('params'):
    if ',' in val:
        param, step = val.split(',')
        fparams[key] = float(param)
    else:
        fparams[key] = float(val)


# load the mass calibration grid
mexprange, zrange, lndM = pickle.load(open(calFile,"rb"))

bigDataDir = Config.get('general','bigDataDirectory')
pzcutoff = Config.getfloat('general','photoZCutOff')

mgrid,zgrid,siggrid = pickle.load(open(bigDataDir+"szgrid_"+expName+"_"+gridName+ "_v" + version+".pkl",'rb'))


assert np.all(mgrid==mexprange)
assert np.all(zrange==zgrid)


saveId = expName + "_" + gridName + "_" + calName + "_v" + version

from orphics.io import dict_from_section, list_from_config
constDict = dict_from_section(Config,'constants')
clusterDict = dict_from_section(Config,'cluster_params')
beam = list_from_config(Config,expName,'beams')
noise = list_from_config(Config,expName,'noises')
freq = list_from_config(Config,expName,'freqs')
lknee = list_from_config(Config,expName,'lknee')[0]
alpha = list_from_config(Config,expName,'alpha')[0]
fsky = Config.getfloat(expName,'fsky')

massMultiplier = Config.getfloat('general','mass_calib_factor')

clttfile = Config.get('general','clttfile')

# get s/n q-bins
qs = list_from_config(Config,'general','qbins')
qspacing = Config.get('general','qbins_spacing')
if qspacing=="log":
    qbins = np.logspace(np.log10(qs[0]),np.log10(qs[1]),int(qs[2])+1)
elif qspacing=="linear":
    qbins = np.linspace(qs[0],qs[1],int(qs[2])+1)
else:
    raise ValueError



cc = ClusterCosmology(fparams,constDict,clTTFixFile=clttfile)
HMF = Halo_MF(cc,mexprange,zrange)

assert numcores==(HMF.zarr.size+1), "ERROR: Need " + str((HMF.zarr.size+1))+" cores."

HMF.sigN = siggrid.copy()
SZProf = SZ_Cluster_Model(cc,clusterDict,rms_noises = noise,fwhms=beam,freqs=freq,lknee=lknee,alpha=alpha)

h = 0.05
# s80, As = getA(fparams,constDict,zrange,kmax=11.)
# s8zs = As*s80

if rank==0:
    dNFid_dmzq = HMF.N_of_mqz_SZ(lndM*massMultiplier,qbins,SZProf)
    np.save(bigDataDir+"N_mzq_"+saveId+"_fid_sigma8",getNmzq(dNFid_dmzq,mgrid,zrange,qbins))

    origPk = HMF.pk.copy()

    
    print("Calculating derivatives for overall power ...")
    HMF.pk = origPk.copy()
    HMF.pk[:,:] *= (1.+old_div(h,2.))**2. 
    dNUp_dmqz = HMF.N_of_mqz_SZ(lndM*massMultiplier,qbins,SZProf)
    Nup = getNmzq(dNUp_dmqz,mgrid,zrange,qbins)

    HMF.pk = origPk.copy()
    HMF.pk[:,:] *= (1.-old_div(h,2.))**2.
    dNDn_dmqz = HMF.N_of_mqz_SZ(lndM*massMultiplier,qbins,SZProf)
    Ndn = getNmzq(dNDn_dmqz,mgrid,zrange,qbins)


    dNdp = old_div((Nup-Ndn),h)

    param = "S8All"

    np.save(bigDataDir+"Nup_mzq_"+saveId+"_"+param,Nup)
    np.save(bigDataDir+"Ndn_mzq_"+saveId+"_"+param,Ndn)
    np.save(bigDataDir+"dNdp_mzq_"+saveId+"_"+param,dNdp)
    
    

else:    

    origPk = HMF.pk.copy()

    
    i = rank-1
    print("Calculating derivatives for redshift ", HMF.zarr[i])
    HMF.pk = origPk.copy()
    HMF.pk[i,:] *= (1.+old_div(h,2.))**2. #((1.+h/2.)*s8zs[i])**2./s8zs[i]**2.
    dNUp_dmqz = HMF.N_of_mqz_SZ(lndM,qbins,SZProf)
    Nup = getNmzq(dNUp_dmqz,mgrid,zrange,qbins)

    HMF.pk = origPk.copy()
    HMF.pk[i,:] *= (1.-old_div(h,2.))**2.
    dNDn_dmqz = HMF.N_of_mqz_SZ(lndM,qbins,SZProf)
    Ndn = getNmzq(dNDn_dmqz,mgrid,zrange,qbins)


    dNdp = old_div((Nup-Ndn),h)

    param = "S8Z"+str(i)

    np.save(bigDataDir+"Nup_mzq_"+saveId+"_"+param,Nup)
    np.save(bigDataDir+"Ndn_mzq_"+saveId+"_"+param,Ndn)
    np.save(bigDataDir+"dNdp_mzq_"+saveId+"_"+param,dNdp)
    


    
