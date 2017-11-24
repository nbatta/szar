import numpy as np
from szar.counts import ClusterCosmology,Halo_MF
import emcee

from orphics.tools.io import dictFromSection, listFromConfig
from ConfigParser import SafeConfigParser
import cPickle as pickle
import matplotlib.pyplot as plt
import nemo
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


class clusterLike(iniFile,expName,gridname):
    def __init__(self):
        
        Config = SafeConfigParser()
        Config.optionxform=str
        Config.read(iniFile)
        bigDataDir = Config.get('general','bigDataDirectory')
        self.clttfile = Config.get('general','clttfile')
        self.constDict = dictFromSection(Config,'constants')
        version = Config.get('general','version')
        
        self.mgrid,self.zgrid,siggrid = pickle.load(open(bigDataDir+"szgrid_"+expName+"_"+gridName+ "_v" + version+".pkl",'rb'))
        self.cc = ClusterCosmology(fparams,constDict,clTTFixFile=clttfile)
        self.HMF = Halo_MF(cc,mgrid,zgrid)

        def lnprior(self,theta):
        a1,a2 = theta
        if  1 < a1 < 5 and  1 < a2 < 2:
            return 0
        return -np.inf

    def lnlike(self,fparams,clustsz,nemo):
        self.cc = ClusterCosmology(fparams,self.constDict,clTTFixFile=self.clttfile)
        self.HMF = Halo_MF(cc,self.mgrid,self.zgrid)

        Ntot = 1.
        Nind = 0
        for i in xrange(len(clustsz)):
            N_per = 1.
            Nind = Nind + np.log(N_per) 
        return -Ntot * Nind

    def lnprob(self,theta, inter, mthresh, zthresh):
        lp = self.lnprior(theta, mthresh, zthresh)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(theta, inter)

