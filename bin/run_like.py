import numpy as np
import szar.likelihood as lk
import matplotlib.pyplot as plt
from scipy import stats
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
print (m_nmap)

LgY = np.arange(-6,-3,0.05)

#timing test NB's macbook pro

#5e-5 seconds / per call
start = time.time()
blah = CL.Y_erf(10**LgY,m_nmap)
print (time.time() - start)
#blah2 = stats.norm.sf(m_nmap,loc = 10**LgY,scale=m_nmap/CL.qmin)
blah2 = 1. - stats.norm.sf(10**LgY,loc = m_nmap,scale=m_nmap/CL.qmin)

for i in range(len(blah)):
    print blah[i],blah2[i],blah[i]/blah2[i]

#1e-4 seconds
#start = time.time()
#blah = CL.P_Yo(LgY,CL.mgrid,0.5)
#print (time.time() - start)

#3e-3 seconds / per call
#start = time.time()
#blah = CL.P_of_SN(LgY,CL.mgrid,0.5,m_nmap)
#print (time.time() - start)

#5e-2 seconds / per call
#start = time.time()
#blah = CL.PfuncY(m_nmap,CL.mgrid,CL.zgrid)
#print (time.time() - start)

#start = time.time()
#blah = CL.PfuncY(m_nmap,CL.HMF.M,CL.HMF.zarr)
#print (time.time() - start)
#print (blah)

area_rads = 987.5/41252.9612

counts = 0.
start = time.time()

count_temp,bin_edge =np.histogram(np.log10(nmap[nmap>0]),bins=20)

frac_of_survey = count_temp*1.0 / np.sum(count_temp)
thresh_bin = 10**((bin_edge[:-1] + bin_edge[1:])/2.)

for i in range(len(frac_of_survey)):
    counts += CL.Ntot_survey(area_rads*frac_of_survey[i],thresh_bin[i])
print (time.time() - start)
print (counts)

