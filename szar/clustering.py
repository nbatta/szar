import numpy as np
from orphics.cosmology import Cosmology
from szar.counts import ClusterCosmology,Halo_MF
from szar.szproperties import SZ_Cluster_Model
import tinker as tinker
from configparser import SafeConfigParser
from orphics.io import dict_from_section

class clustering:
    def __init__(self,iniFile):
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
        self.SZProp = SZ_Cluster_Model(self.cc,self.clusterDict,rms_noises = noise,fwhms=beam,freqs=freq,lknee=lknee,alpha=alpha)

    def ntilde(self):
        dndm_SZ = self.HMF.dn_dmz_SZ(self.SZprop)
        ans = np.trapz(dndm_SZ,dx=np.diff(self.HMF.M200),axis=0)
        return ans

    def b_eff_z(self):
        ''' 
        effective linear bias wieghted by number density
        '''
        nbar = self.ntilde()

        z_arr = self.HMF.zarr
        dndm_SZ = self.HMF.dn_dmz_SZ(self.SZprop)
        
        R = tinker.radius_from_mass(self.HMF.M200,self.cc.rhoc0om)
        sigsq = tinker.sigma_sq_integral(R, self.HMF.pk, self.HMF.kh)

        blin = tinker.tinker_bias(sigsq,200.)
        beff = np.trapz(dndm_SZ*blin,dx=np.diff(self.HMF.M200),axis=0) / nbar

        return beff
        
    def Norm_Sfunc(self,fsky):
        z_arr = self.HMF.zarr
        nbar = self.ntilde()
        ans = np.trapz(self.HMF.dVdz*nbar**2,dx=np.diff(z_arr))
        return ans*4.*np.pi*fsky

    def power_spec(self,z_arr):
        
        k = self.HMF.kh
        pk = self.HMF.pk
        ans = 1 
        return ans

