import matplotlib
matplotlib.use('Agg')
import camb
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import sys, os, time
from szlib.szcounts import ClusterCosmology,SZ_Cluster_Model,Halo_MF
from orphics.tools.io import Plotter,dictFromSection,listFromConfig
from ConfigParser import SafeConfigParser 

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



# accuracy params
dell=10
pmaxN=5
numps=1000
tmaxN=5
numts=1000

cosmoDict = dictFromSection(Config,cosmologyName)
constDict = dictFromSection(Config,'constants')
clusterDict = dictFromSection(Config,clusterParams)
cc = ClusterCosmology(cosmoDict,constDict,clTTFixFile = "data/cltt_lensed_Feb18.txt")#,skipCls=True)

zs = np.arange(0.5,3.0,0.2)
Mexp = np.arange(14.0,15.,0.5)

# zs = np.arange(0.1,3.0,0.1)
# Mexp = np.arange(13.0,15.7,0.2)



hmf = Halo_MF(cc,Mexp,zs)


SZProf = SZ_Cluster_Model(cc,clusterDict,rms_noises = noise,fwhms=beam,freqs=freq,lknee=lknee,alpha=alpha)

fsky = 0.4

N1 = hmf.N_of_z(fsky)
N2 = hmf.N_of_z_SZ(fsky,SZProf)

from orphics.tools.io import Plotter
pl = Plotter(scaleY='log')
pl.add(zs,N1)
pl.add(zs,N2)


pl.done("output/Ns.png")
