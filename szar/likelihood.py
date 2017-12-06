import numpy as np
from szar.counts import ClusterCosmology,Halo_MF
import emcee
from nemo import simsTools
from ConfigParser import SafeConfigParser
from orphics.tools.io import dictFromSection
import cPickle as pickle
import matplotlib.pyplot as plt

#import time


class clusterLike:
    def __init__(self,iniFile,expName,gridName,parDict,nemoOutputDir):
        
        Config = SafeConfigParser()
        Config.optionxform=str
        Config.read(iniFile)

        self.fparams = {}
        for (key, val) in Config.items('params'):
            if ',' in val:
                param, step = val.split(',')
                self.fparams[key] = float(param)
            else:
                self.fparams[key] = float(val)

        bigDataDir = Config.get('general','bigDataDirectory')
        self.clttfile = Config.get('general','clttfile')
        self.constDict = dictFromSection(Config,'constants')
        version = Config.get('general','version')
        
        self.mgrid,self.zgrid,siggrid = pickle.load(open(bigDataDir+"szgrid_"+expName+"_"+gridName+ "_v" + version+".pkl",'rb'))
        self.cc = ClusterCosmology(self.fparams,self.constDict,clTTFixFile=self.clttfile)
        self.HMF = Halo_MF(self.cc,self.mgrid,self.zgrid)

        self.diagnosticsDir=nemoOutputDir+"diagnostics" 
        filteredMapsDir=nemoOutputDir+"filteredMaps"
        self.tckQFit=simsTools.fitQ(parDict, self.diagnosticsDir, filteredMapsDir)

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

#Functions from NEMO
#y0FromLogM500(log10M500, z, tckQFit, tenToA0 = 4.95e-5, B0 = 0.08, Mpivot = 3e14, sigma_int = 0.2)
#fitQ(parDict, diagnosticsDir, filteredMapsDir)

    #self.diagnosticsDir=nemoOutputDir+os.path.sep+"diagnostics"
    
    #filteredMapsDir=nemoOutputDir+os.path.sep+"filteredMaps"
    #self.tckQFit=simsTools.fitQ(parDict, self.diagnosticsDir, filteredMapsDir)
