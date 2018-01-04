import numpy as np
import szar.likelihood as lk
import matplotlib.pyplot as plt
from scipy import stats
from configparser import SafeConfigParser
from orphics.tools.io import dictFromSection
from szar.counts import ClusterCosmology,Halo_MF
import emcee
import time

iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)
bigDataDir = Config.get('general','bigDataDirectory')
clttfile = Config.get('general','clttfile')
constDict = dictFromSection(Config,'constants')
version = Config.get('general','version')
expName = "S4-1.5-paper" #S4-1.0-CDT"
gridName = "grid-owl2" #grid-owl2"
#_S4-1.5-paper_grid-owl2_v0.6.p
fparams = {}
for (key, val) in Config.items('params'):
    if ',' in val:
        param, step = val.split(',')
        fparams[key] = float(param)
    else:
        fparams[key] = float(val)
        
nemoOutputDir = '/gpfs01/astro/workarea/msyriac/data/depot/SZruns/ACTdata/' #/Users/nab/Desktop/Projects/ACTPol_Cluster_Like/ACTdata/'
nemoOutputDirOut = '/gpfs01/astro/workarea/msyriac/data/depot/SZruns/ACTdata_out/'
pardict = nemoOutputDir + 'equD56.par'
noise_file = 'RMSMap_Arnaud_M2e14_z0p4.fits'
CL = lk.clusterLike(iniFile,expName,gridName,pardict,nemoOutputDir,noise_file)

diagnosticsDir = '/gpfs01/astro/workarea/msyriac/data/depot/SZruns/ACTdata_out/'

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


count_temp,bin_edge =np.histogram(np.log10(nmap[nmap>0]),bins=20)

frac_of_survey = count_temp*1.0 / np.sum(count_temp)
thresh_bin = 10**((bin_edge[:-1] + bin_edge[1:])/2.)

parlist = ['omch2','ombh2','H0','As','ns','tau','massbias','yslope','scat']
parvals = [0.1194,0.022,67.0,2.2e-09,0.96,0.06,0.80,0.08,0.2]

#print fparams 

params= CL.alter_fparams(fparams,parlist,parvals)

#print params

start = time.time()
int_cc = ClusterCosmology(params,CL.constDict,clTTFixFile=CL.clttfile) # internal HMF call
print ('CC',time.time() - start)
start = time.time()
int_HMF = Halo_MF(int_cc,CL.mgrid,CL.zgrid)
print ('HMF',time.time() - start)

cluster_prop = np.array([CL.clst_z,CL.clst_zerr,CL.clst_y0,CL.clst_y0err])
cluster_prop2 = np.array([CL.clst_z,CL.clst_zerr,CL.clst_y0*1e-4,CL.clst_y0err*1e-4])
print cluster_prop.shape

dndm_int = int_HMF.inter_dndm(200.)

start = time.time()
print np.log(CL.Prob_per_cluster(int_HMF,cluster_prop2[:,0],dndm_int,params))
print ('per cluster spec z',time.time() - start)

c_z, c_zerr, c_y, c_yerr = cluster_prop[:,0]
c_y *= 1e-4
c_yerr *= 1e-4
#print c_z, c_zerr, c_y, c_yerr
#print int_HMF.zarr

start = time.time()
print np.log(CL.Prob_per_cluster(int_HMF,cluster_prop2[:,1],dndm_int,params))
print ('per cluster photo z',time.time() - start)

#mind = 50

#print c_y,c_yerr
#print 'prob check 0',CL.Y_prob(c_y,CL.LgY,c_yerr)
#print 'prob check 1',CL.Pfunc_per(CL.HMF.M.copy(),c_z,c_y, c_yerr,params)
#print CL.P_of_Y_per(CL.HMF.M.copy())[mind]
#print CL.Pfunc_per(CL.HMF.M.copy(),c_z,c_y,c_yerr,params)
#print 'MF interpol check',dndm_int(c_z,CL.HMF.M.copy())[mind,0]/int_HMF.dn_dM(int_HMF.M200,200.)[mind,2]
#print 'Mass check',int_HMF.cc.Mass_con_del_2_del_mean200(CL.HMF.M.copy(),500,c_z)[mind] / CL.HMF.M.copy()[mind]

#start = time.time()
#for i in range(len(frac_of_survey)):
#    counts += CL.Ntot_survey(int_HMF,area_rads*frac_of_survey[i],thresh_bin[i],params)
#print ('Ntot loop',time.time() - start)
#print (counts)

start = time.time()
print CL.lnlike(parvals,parlist)
print ('Ln Like Func',time.time() - start)

#parlist = ['omch2','As','tau','massbias','scat','yslope']
#parvals = [0.1194,2.2e-09,0.06,1.0,0.15,0.08]

#params_test= CL.alter_fparams(fparams,parlist,parvals)

priorlist = ['tau','ns','H0','massbias','scat']
prioravg = np.array([0.06,0.96,67,0.8,0.2])
priorwth = np.array([0.01,0.01,3,0.12,0.05])
priorvals = np.array([prioravg,priorwth])

print CL.lnprior(parvals,parlist,priorvals,priorlist)

##test likelihood
#Ndim, nwalkers = 5 , 10
#P0 = np.array(parvals)

#pos = [P0 + P0*1e-1*np.random.randn(Ndim) for i in range(nwalkers)]

#start = time.time()
#sampler = emcee.EnsembleSampler(nwalkers,Ndim,CL.lnlike,args =(parlist))
#sampler.run_mcmc(pos,2)
#print (time.time() - start)  

#emcee_run_lnlk = sampler.lnprobability
#emcee_run_chain =  sampler.chain[:,:,:].reshape((-1,Ndim))

#print CL.lnlike(emcee_run_chain[0],parlist,np.array([1.,1.]))
#print CL.lnlike(emcee_run_chain[10],parlist,np.array([1.,1.]))
#print (emcee_run_lnlk)
#print (emcee_run_chain)


#{'beta_ym': 0.0, 'Ysig': 0.127, 'betaYsig': 0.0, 'Y_star': 2.42e-10, 'wa': 0.0, 'tau': 0.06, 'b_wl': 1.0, 'H0': 67.0, 'S8All': 0.8, 'mnu': 0.06, 'alpha_ym': 1.79, 'As': 2.2e-09, 'sigR': 0.75, 'omch2': 0.1194, 'gammaYsig': 0.0, 'w0': -1.0, 'rho_corr': 0.0, 'ns': 0.96, 'gamma_ym': 0.0, 'ombh2': 0.022, 'b_ym': 0.8}
