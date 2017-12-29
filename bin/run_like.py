import numpy as np
import szar.likelihood as lk
import matplotlib.pyplot as plt
from scipy import stats
from configparser import SafeConfigParser
from orphics.tools.io import dictFromSection
from szar.counts import ClusterCosmology,Halo_MF

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

fparams = {}
for (key, val) in Config.items('params'):
    if ',' in val:
        param, step = val.split(',')
        fparams[key] = float(param)
    else:
        fparams[key] = float(val)
        
nemoOutputDir = '/Users/nab/Desktop/Projects/ACTPol_Cluster_Like/ACTdata/'
nemoOutputDirOut = '/Users/nab/Desktop/Projects/ACTPol_Cluster_Like/'
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
blah2 = 1. - stats.norm.sf(10**LgY,loc = m_nmap*CL.qmin,scale=m_nmap)
blah3 = 1. - stats.norm.sf(10**LgY,loc = m_nmap,scale=m_nmap/CL.qmin)

#for i in range(len(blah)):
#    print blah[i],blah2[i],blah[i]/blah2[i]


thk = 3
plt.figure(figsize=(10,8))
plt.rc('axes', linewidth=thk)
plt.tick_params(size=14,width=thk,labelsize = 16)
plt.xlabel(r'$\mathrm{Log}_{10}Y$', fontsize=32,weight='bold')
plt.ylabel(r'$P(Y)$', fontsize=32,weight='bold')
plt.plot(LgY,blah,linewidth=thk)
plt.plot(LgY,blah2,'--',linewidth=thk/2.)
plt.plot(LgY,blah3,'--',linewidth=thk)
plt.plot(np.log10([m_nmap,m_nmap]),[0,1],'--k',linewidth=thk)
plt.plot(np.log10([CL.qmin*m_nmap,CL.qmin*m_nmap]),[0,1],'--r',linewidth=thk)
plt.savefig(nemoOutputDirOut+'P_Y_erf_comp_MJH.png',bbox_inches='tight',format='png')

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

parlist = ['omch2','ombh2','H0','As','ns','tau','massbias','yslope','scat']
parvals = [0.1194,0.022,67.0,2.2e-09,0.96,0.06,1.0,1.08,0.2]

#print fparams 

params= CL.alter_fparams(fparams,parlist,parvals)

#print params

int_cc = ClusterCosmology(fparams,CL.constDict,clTTFixFile=CL.clttfile) # internal HMF call
int_HMF = Halo_MF(int_cc,CL.mgrid,CL.zgrid)

for i in range(len(frac_of_survey)):
    counts += CL.Ntot_survey(int_HMF,area_rads*frac_of_survey[i],thresh_bin[i],fparams)
print (time.time() - start)
print (counts)

#print fparams


#{'beta_ym': 0.0, 'Ysig': 0.127, 'betaYsig': 0.0, 'Y_star': 2.42e-10, 'wa': 0.0, 'tau': 0.06, 'b_wl': 1.0, 'H0': 67.0, 'S8All': 0.8, 'mnu': 0.06, 'alpha_ym': 1.79, 'As': 2.2e-09, 'sigR': 0.75, 'omch2': 0.1194, 'gammaYsig': 0.0, 'w0': -1.0, 'rho_corr': 0.0, 'ns': 0.96, 'gamma_ym': 0.0, 'ombh2': 0.022, 'b_ym': 0.8}
