import numpy as np
from szar.counts import ClusterCosmology,Halo_MF
from orphics.tools.io import dictFromSection, listFromConfig
from ConfigParser import SafeConfigParser
import cPickle as pickle
import matplotlib.pyplot as plt
import emcee
import time

#initial setup

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

mgrid,zgrid,siggrid = pickle.load(open(bigDataDir+"szgrid_"+expName+"_"+gridName+ "_v" + version+".pkl",'rb'))

cc = ClusterCosmology(fparams,constDict,clTTFixFile=clttfile)
HMF = Halo_MF(cc,mgrid,zgrid)

bla = HMF.N_of_Mz(HMF.M200,200)
print HMF.M[220]
print HMF.zarr[18]

print bla[220][18]

blah = HMF.sample_mf(200)#(1.8,4e14,bla)

print blah(HMF.zarr[18],HMF.M[220])

#plt.imshow(np.log10(bla))
#plt.colorbar()
#plt.savefig('default.png', bbox_inches='tight',format='png')

print blah(1.58674232,10**15.38257871)
print blah(1.58,10**15.3)
print np.log10(blah(1.01954414, 10**15.6462284)), np.log10(blah(0.15,9e13))

def lnprior(theta):
    a1,a2temp = theta
    a2 = 10**a2temp
    if  0.2 < a1 < 1.95 and  4e14 < a2 < 4e15:
        return 0
    return -np.inf

def lnlike(theta,inter):
    a1,a2temp = theta
    a2 = 10**a2temp
    return np.log(inter(a1,a2)/inter(0.15,9e13))

def lnprob(theta, inter):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, inter)

Ndim, nwalkers = 2 , 100

P0 = np.array([1.,15.5])

pos = [P0 + P0*2e-2*np.random.randn(Ndim) for i in range(nwalkers)]

start = time.clock()
sampler = emcee.EnsembleSampler(nwalkers,Ndim,lnprob, args =[blah] )
sampler.run_mcmc(pos,1000)
elapsed1 = (time.clock() - start)
print elapsed1

samples = sampler.chain[:,50:,:].reshape((-1,Ndim))

print np.shape(samples)

plt.plot(samples[:,0],samples[:,1],'x')
plt.savefig('default2.png', bbox_inches='tight',format='png') 
