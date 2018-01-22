import numpy as np
from szar.counts import ClusterCosmology,Halo_MF
from szar.szproperties import SZ_Cluster_Model
#from orphics import dict_from_section, list_from_config
from orphics.io import dict_from_section
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
constDict = dict_from_section(Config,'constants')
clusterDict = dict_from_section(Config,'cluster_params')
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
samples = HMF.mcsample_mf(200.,1000,mthresh=[np.log10(3e14),np.log10(7e15)])
elapsed1 = (time.clock() - start)
print elapsed1
print len(samples)

nclust = 100
ids = np.random.randint(len(samples)/2.,size=nclust)

plt.plot(samples[:,0],samples[:,1],'x')
plt.plot(samples[ids,0],samples[ids,1],'o')
plt.savefig('default_mf.png', bbox_inches='tight',format='png') 

sys.exit(0)

experimentName = expName

beams = list_from_config(Config,experimentName,'beams')
noises = list_from_config(Config,experimentName,'noises')
freqs = list_from_config(Config,experimentName,'freqs')
lmax = int(Config.getfloat(experimentName,'lmax'))
lknee = list_from_config(Config,experimentName,'lknee')[0]
alpha = list_from_config(Config,experimentName,'alpha')[0]
fsky = Config.getfloat(experimentName,'fsky')

SZProp = SZ_Cluster_Model(clusterCosmology=cc,clusterDict=clusterDict,rms_noises = noises,fwhms=beams,freqs=freqs,lmax=lmax,lknee=lknee,alpha=alpha)

MM = 10**(samples[ids,1])
zz = samples[ids,0]

Y = SZProp.Y_M(MM,zz) * np.random.normal(1,0.2,nclust)

ML = MM * np.random.normal(1,0.2,nclust)

plt.loglog(ML,Y,'o') 
#plt.savefig('default2.png', bbox_inches='tight',format='png')

#Z = MM
