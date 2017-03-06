import matplotlib
matplotlib.use('Agg')
import camb
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import sys, os, time
from szlib.szcounts import ClusterCosmology,SZ_Cluster_Model,Halo_MF,getTotN
from orphics.tools.io import Plotter,dictFromSection,listFromConfig
from ConfigParser import SafeConfigParser 
import cPickle as pickle
from orphics.tools.io import Plotter
from orphics.analysis.flatMaps import interpolateGrid

clusterParams = 'LACluster' # from ini file
cosmologyName = 'LACosmology' # from ini file
experimentName = "LATest"

iniFile = "input/params.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)


beam = listFromConfig(Config,experimentName,'beams')
noise = listFromConfig(Config,experimentName,'noises')
freq = listFromConfig(Config,experimentName,'freqs')
lmax = int(Config.getfloat(experimentName,'lmax'))
lknee = Config.getfloat(experimentName,'lknee')
alpha = Config.getfloat(experimentName,'alpha')
fsky = Config.getfloat(experimentName,'fsky')


cosmoDict = dictFromSection(Config,cosmologyName)
constDict = dictFromSection(Config,'constants')
clusterDict = dictFromSection(Config,clusterParams)
cc = ClusterCosmology(cosmoDict,constDict,clTTFixFile = "data/cltt_lensed_Feb18.txt")#,skipCls=True)


mfile = "data/S4-7mCMB_all.pkl"
minrange, zinrange, lndM = pickle.load(open(mfile,'rb'))


# zs = np.arange(0.5,3.0,0.5)
# Mexp = np.arange(13.5,15.7,0.5)

zs = np.arange(0.1,3.0,0.1)
Mexp = np.arange(13.0,15.7,0.1)

outmerr = interpolateGrid(lndM,minrange,zinrange,Mexp,zs,regular=False,kind="cubic",bounds_error=False,fill_value=np.inf)





hmf = Halo_MF(cc,Mexp,zs)


SZProf = SZ_Cluster_Model(cc,clusterDict,rms_noises = noise,fwhms=beam,freqs=freq,lknee=lknee,alpha=alpha)

fsky = 0.4

N1 = hmf.N_of_z()*fsky
N2 = hmf.N_of_z_SZ(SZProf)*fsky

pl = Plotter(scaleY='log')
pl.add(zs,N1)
pl.add(zs,N2)




sn,ntot = hmf.Mass_err(fsky,outmerr,SZProf)

print sn
print ntot


q_arr = np.logspace(np.log10(6.),np.log10(500.),40)
dnqmz = hmf.N_of_mqz_SZ(outmerr,q_arr,SZProf)


N,Nofz = getTotN(dnqmz,10**Mexp,zs,q_arr,returnNz=True)

print N

pl.add(zs,Nofz)

pl.done("output/Ns.png")
