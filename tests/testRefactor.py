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

outDir = os.environ['WWW']

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

zs = np.arange(0.1,3.0,0.3)
Mexp = np.arange(13.0,15.7,0.3)

zs = np.insert(zs,0,0)

outmerr = interpolateGrid(lndM,minrange,zinrange,Mexp,zs[1:],regular=False,kind="cubic",bounds_error=False,fill_value=np.inf)





hmf = Halo_MF(cc)


SZProf = SZ_Cluster_Model(cc,clusterDict,rms_noises = noise,fwhms=beam,freqs=freq,lknee=lknee,alpha=alpha)

fsky = 0.4

N1 = hmf.N_of_z(Mexp,zs)*fsky*4*np.pi

#hmf.sigN = np.loadtxt("temp.txt")

N2 = hmf.N_of_z_SZ(Mexp,zs,beam,noise,freq,clusterDict,lknee,alpha)*fsky*hmf.dVdz(zs)[1:]*4*np.pi



pl = Plotter()
pl.plot2d(hmf.sigN)
pl.done(outDir+"signMaster.png")

pl = Plotter(scaleY='log')
pl.add(zs[1:],N1)
pl.add(zs[1:],N2)

Ntot1 = np.trapz(N2,zs[1:])
print Ntot1


sn,ntot = hmf.Mass_err(fsky,outmerr,Mexp,zs,beam,noise,freq,clusterDict,lknee,alpha)

print ntot



q_arr = np.logspace(np.log10(6.),np.log10(500.),64)

print zs
dnqmz = hmf.N_of_mqz_SZ(outmerr,zs,Mexp,q_arr,beam,noise,freq,clusterDict,lknee,alpha)


N,Nofz = getTotN(dnqmz,Mexp,zs[1:],q_arr,returnNz=True)

print N*fsky

pl.add(zs[1:],Nofz*fsky,label="mqz")
pl.legendOn()

pl.done(outDir+"NsMaster.png")


nnoq = np.trapz(dnqmz,q_arr,axis=2)*fsky
pl = Plotter()
pl.plot2d(nnoq)
pl.done(outDir+"ngridMaster.png")
