import camb
import numpy as np

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

cosmoDict = dictFromSection(Config,cosmologyName)
constDict = dictFromSection(Config,'constants')
clusterDict = dictFromSection(Config,clusterParams)
cc = ClusterCosmology(cosmoDict,constDict,lmax)

zz = 0.5
MM = 5e14 #M/h

SZ = SZ_Cluster_Model(cc,clusterDict,lmax=lmax )

var = SZ.quickVar(MM,zz)
print np.sqrt(var)
print SZ.Y_M(MM,zz)

ells = np.arange(2,6000,10)

nl = SZ.noise_func(ells,beam[0],noise[0])

np.savetxt('/Users/nab/Desktop/Projects/SO_forecasts/test_noise.txt',np.transpose([ells,nl]))

