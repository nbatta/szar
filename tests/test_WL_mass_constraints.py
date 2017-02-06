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
experimentName = "AdvAct"
#fileFunc = None
fileFunc = lambda M,z:"data/"+experimentName+"_m"+str(M)+"z"+str(z)+".txt"


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

mass_err_file = Config.get(experimentName,'mass_err')
mass_err = np.loadtxt(mass_err_file)

zbin_temp = np.arange(0.1,2.01,0.05)
zbin = np.insert(zbin_temp,0,0.0)

HMF = Halo_MF(clusterCosmology=cc)

errs,Ntot = HMF.Mass_err(mass_err,zbin,beam,noise,freq,clusterDict,lknee,alpha,fileFunc)

print np.sqrt(errs)
print Ntot

#HSC_mass = np.loadtxt('input/HSC_DeltalnM_z0_z1.txt',unpack=True)
#HSC_mass = np.transpose(HSC_mass)

#print np.shape(HSC_mass), np.shape(dndzdm)
