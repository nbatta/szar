import numpy as np
from orphics.cosmology import Cosmology
from szar.counts import ClusterCosmology,Halo_MF
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
        version = Config.get('general','version')

        self.mgrid,self.zgrid,siggrid = pickle.load(open(bigDataDir+"szgrid_"+expName+"_"+gridName+ "_v" + version+".pkl",'rb'))  

        self.cc = ClusterCosmology(self.fparams,self.constDict,clTTFixFile=self.clttfile)
        self.HMF = Halo_MF(self.cc,self.mgrid,self.zgrid)

    def b_eff(self):
        ''' 
        effective linear bias wieghted by number density
        '''
        nbar = self.HMF.nz()

        z_arr = self.HMF.zarr
        dndm = self.HMF.dn_dm(self.M200,200.)
        
        R = tinker.radius_from_mass(self.HMF.M200,self.cc.rhoc0om)
        sigsq = tinker.sigma_sq_integral(R, self.HMF.pk, self.HMF.kh)

        blin = tinker.tinker_bias(sigsq,200.)
        beff = np.trapz(dndm*blin,dx=np.diff(self.HMF.M200),axis=0)

        return beff
        

    def power_spec(self,z_arr):
        
        k = self.HMF.kh
        pk = self.HMF.pk
        ans = 1 
        return ans

