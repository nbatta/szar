import numpy as np
from orphics.cosmology import Cosmology
from szar.counts import ClusterCosmology,Halo_MF



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

    def b_eff(self,):
        ''' 
        Number density! Change this
        '''
        nbar = self.HMF.N_of_z()

        z_arr = self.zarr
        dn_dzdm = self.N_of_Mz(self.M200,200.)
        N_z = np.zeros(z_arr.size)
        blin = 1.
        for i in range(z_arr.size):
            N_z[i] = np.trapz(dn_dzdm[:,i]*blin,dx=np.diff(self.M200_edges[:,i]))

        beff = N_z*4.*np.pi
        return beff
        

    def power_spec(self,z_arr):
        
        k = self.kh
        pk = self.pk
        ans = 1 
        return ans

