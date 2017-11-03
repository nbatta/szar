import unittest
from orphics.tools.io import dictFromSection, listFromConfig
from configparser import SafeConfigParser 
import numpy as np

class TestHMF(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestHMF, self).__init__(*args, **kwargs)

        iniFile = "tests/unitTests.ini"
        Config = SafeConfigParser()
        Config.optionxform=str
        Config.read(iniFile)


        self.testname = Config.get('general','testName')

        ms = listFromConfig(Config,self.testname,'mexp')
        self.marray = np.arange(ms[0],ms[1],ms[2])
        self.zs = listFromConfig(Config,self.testname,'zs')

    
    def test_Mass_Err(self):
        print((self.marray))
        print((self.zs))
    def Halo_Tinker_test(self):
        
        #define parameters delta, M and z
        z_arr = np.array([0,0.8,1.6])
        M = 10**np.arange(10., 16, .1)
        delts = z_arr*0 + 200.
        delts_8 = z_arr*0 + 800.
        delts_32 = z_arr*0 + 3200.
    
        #get p of k and s8 
        kh, pk = self._pk(z_arr)
        # dn_dlogM from tinker
        N = dn_dlogM(M,z_arr,self.cc.rhoc0om,delts,kh,pk[:1,:])
        N_8 = dn_dlogM(M,z_arr,self.cc.rhoc0om,delts_8,kh,pk[:1,:])
        N_32 = dn_dlogM(M,z_arr,self.cc.rhoc0om,delts_32,kh,pk[:1,:])
        
        #plot tinker values
        pl = Plotter()
        pl._ax.set_ylim(-3.6,-0.8)
        pl.add(np.log10(M),np.log10(N[:,0]*M/(self.cc.rhoc0om)),color='black')
        pl.add(np.log10(M),np.log10(N_8[:,0]*M/(self.cc.rhoc0om)),color='black')
        pl.add(np.log10(M),np.log10(N_32[:,0]*M/(self.cc.rhoc0om)),color='black')
        pl.done("output/tinkervals.png")
        
        return f

    def Y_M_test(self,MM,zz):
        DA_z = self.cc.results.angular_diameter_distance(zz) 
        Y_star = 0.6456
        alpha_ym = 1.79
        b_ym = 0.8
        beta_ym = 0.66666 
        gamma_ym = 0 
        beta_fac = 1.
        ans = Y_star*((b_ym)*MM/(self.cc.H0/100.)/6e14)**alpha_ym *(1e-4/DA_z**2) * beta_fac * self.cc.E_z(zz) ** (2./3.) * (1. + zz)**gamma_ym
        #print (0.01/DA_z)**2
        return ans

if __name__ == '__main__':
    unittest.main()
