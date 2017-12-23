import numpy as np
import szar.likelihood as lk
import matplotlib.pyplot as plt
from configparser import SafeConfigParser
from orphics.tools.io import dictFromSection

import time

iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)
bigDataDir = Config.get('general','bigDataDirectory')
clttfile = Config.get('general','clttfile')
constDict = dictFromSection(Config,'constants')
version = Config.get('general','version')
expName = "S4-1.0-CDT"
gridName = "grid-owl2"

#fparams = {}
#for (key, val) in Config.items('params'):
#    if ',' in val:
#        param, step = val.split(',')
#        fparams[key] = float(param)
#    else:
#        fparams[key] = float(val)

nemoOutputDir = '/Users/nab/Desktop/Projects/ACTPol_Cluster_Like/ACTdata/'
pardict = nemoOutputDir + 'equD56.par'
noise_file = 'RMSMap_Arnaud_M2e14_z0p4.fits'
CL = lk.clusterLike(iniFile,expName,gridName,pardict,nemoOutputDir,noise_file)

diagnosticsDir = '/Users/nab/Downloads/countsCheck/equD56-countsCheck/diagnostics/'

nmap = lk.read_MJH_noisemap(nemoOutputDir+noise_file,diagnosticsDir+'areaMask.fits')


print nmap.shape
m_nmap = np.mean(nmap[nmap>0])

LgY = np.arange(-6,-3,0.05)

#timing test NB's macbook pro

#5e-5 seconds / per call
#start = time.time()
#blah = CL.Y_erf(10**LgY,m_nmap)
#print (time.time() - start)

#1e-4 seconds
#start = time.time()
#blah = CL.P_Yo(LgY,CL.mgrid,0.5)
#print (time.time() - start)

#3e-3 seconds / per call
start = time.time()
blah = CL.P_of_SN(LgY,CL.mgrid,0.5,m_nmap)
print (time.time() - start)
