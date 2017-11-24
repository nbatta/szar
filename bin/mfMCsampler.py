import numpy as np
from szar.counts import ClusterCosmology,Halo_MF
from szar.szproperties import SZ_Cluster_Model
from orphics.tools.io import dictFromSection, listFromConfig
from ConfigParser import SafeConfigParser
import cPickle as pickle
import matplotlib.pyplot as plt
#import emcee
import time

#initial setup

iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)
bigDataDir = Config.get('general','bigDataDirectory')
clttfile = Config.get('general','clttfile')
constDict = dictFromSection(Config,'constants')
clusterDict = dictFromSection(Config,'cluster_params')
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

start = time.clock()
cc = ClusterCosmology(fparams,constDict,clTTFixFile=clttfile)
elapsed1 = (time.clock() - start)
print elapsed1
start = time.clock()
HMF = Halo_MF(cc,mgrid,zgrid)
elapsed1 = (time.clock() - start)
print elapsed1

start = time.clock()
samples = HMF.mcsample_mf(200.,1000,mthresh=[3e14,7e15])
elapsed1 = (time.clock() - start)
print elapsed1
print len(samples)

nclust = 1000
ids = np.random.randint(len(samples)/2.,size=nclust)

#plt.plot(samples[:,0],samples[:,1],'x')
#plt.plot(samples[ids,0],samples[ids,1],'o')
#plt.savefig('default2.png', bbox_inches='tight',format='png') 

experimentName = expName

beams = listFromConfig(Config,experimentName,'beams')
noises = listFromConfig(Config,experimentName,'noises')
freqs = listFromConfig(Config,experimentName,'freqs')
lmax = int(Config.getfloat(experimentName,'lmax'))
lknee = listFromConfig(Config,experimentName,'lknee')[0]
alpha = listFromConfig(Config,experimentName,'alpha')[0]
fsky = Config.getfloat(experimentName,'fsky')

SZProp = SZ_Cluster_Model(clusterCosmology=cc,clusterDict=clusterDict,rms_noises = noises,fwhms=beams,freqs=freqs,lmax=lmax,lknee=lknee,alpha=alpha)

MM = 10**(samples[ids,1])
zz = samples[ids,0]

Y = SZProp.Y_M(MM,zz) * np.random.normal(1,0.2,nclust)

ML = MM * np.random.normal(1,0.2,nclust)

plt.loglog(ML,Y,'o') 
plt.savefig('default2.png', bbox_inches='tight',format='png')

#Z = MM
