from __future__ import division
from builtins import object
import numpy as np
from scipy.special import j0
#from orphics.cosmology import Cosmology
from szar.counts import ClusterCosmology,Halo_MF
from configparser import SafeConfigParser
from orphics.io import dict_from_section
from szar.clustering import clustering

class pairwise(object):
    def __init__(self,iniFile,kmin=1e-4,kmax=5.,knum=200):

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
        self.constDict = dict_from_section(Config,'constants')
        self.clusterDict = dictFromSection(Config,'cluster_params')
        version = Config.get('general','version')
        beam = listFromConfig(Config,expName,'beams')
        noise = listFromConfig(Config,expName,'noises')
        freq = listFromConfig(Config,expName,'freqs')
        lknee = listFromConfig(Config,expName,'lknee')[0]
        alpha = listFromConfig(Config,expName,'alpha')[0]

        self.mgrid,self.zgrid,siggrid = pickle.load(open(bigDataDir+"szgrid_"+expName+"_"+gridName+ "_v" + version+".pkl",'rb'))

        self.cc = ClusterCosmology(self.fparams,self.constDict,clTTFixFile=self.clttfile)
        self.HMF = Halo_MF(self.cc,self.mgrid,self.zgrid)

        if powerZK is None:
            self.kh, self.pk = self._pk_lin(self.HMF.zarr,kmin,kmax,knum)
        else:
            assert kh is not None
            self.kh = kh
            self.pk = powerZK

    def _pk_lin(self,zarr,kmin,kmax,knum): #Linear PK
        self.cc.pars.set_matter_power(redshifts=zarr, kmax=kmax)
        self.cc.pars.Transfer.high_precision = True

        self.cc.pars.Linear = model.Linear_none
        self.cc.results = camb.get_results(self.cc.pars)

        kh, z, powerZK = self.cc.results.get_matter_power_spectrum(minkh=kmin, maxkh=kmax, npoints = knum)

        return kh, powerZK

    def massWeightedbias(self,q):

        z_arr = self.HMF.zarr
        dndm_SZ = self.HMF.dn_dmz_SZ(self.SZprop)

        R = tinker.radius_from_mass(self.HMF.M200,self.cc.rhoc0om)
        sigsq = tinker.sigma_sq_integral(R, self.HMF.pk, self.HMF.kh)

        blin = tinker.tinker_bias(sigsq,200.)
        #add loop over k for bweight / bnorm
        bweight = np.trapz(dndm*blin**q*self.HMF.M200,dx=np.diff(self.HMF.M200),axis=0)
        bnorm = np.trapz(dndm*self.HMF.M200,dx=np.diff(self.HMF.M200),axis=0)

        ans = bweight/bnorm

        return ans

    def zeta(self,rad):

        k_arr = self.HMF.kh.copy()
        integ = np.trapz(k_arr**2*self.massWeightedbias(2)*j0(k_arr*rad)*self.pk,dx=np.diff(k_arr),axis=0)/ (2.*np.pi**2)

        return integ

    def zetabar(self,rad):

        integ = 3.*np.trapz(rad**2*self.massWeightedbias(1)*self.zeta(rad),dx=np.diff(rad),axis=0)/ rad**3

        return integ

    def meanvel(self):

        Hubble = 1
        fg = 1.
        rad_arr = np.arange(1000) * 0.1
        ans = -2./3.*1./(1. + self.zarr) * Hubble * fg * (rad_arr*self.zetabar(rad_arr)/(1.+self.zeta(rad_arr)))

        return ans
