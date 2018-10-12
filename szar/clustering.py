from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from future import standard_library
standard_library.install_aliases()
from builtins import object
from past.utils import old_div
import numpy as np
#from orphics.cosmology import Cosmology
from szar.counts import ClusterCosmology,Halo_MF
from szar.szproperties import SZ_Cluster_Model
from . import tinker as tinker
from configparser import SafeConfigParser
from orphics.io import dict_from_section,list_from_config
import pickle as pickle
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
#from scipy.integrate import simps

class Clustering(object):
    def __init__(self,iniFile,expName,gridName,version):
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
        self.clusterDict = dict_from_section(Config,'cluster_params')
        #version = Config.get('general','version')
        beam = list_from_config(Config,expName,'beams')
        noise = list_from_config(Config,expName,'noises')
        freq = list_from_config(Config,expName,'freqs')
        lknee = list_from_config(Config,expName,'lknee')[0]
        alpha = list_from_config(Config,expName,'alpha')[0]

        self.mgrid,self.zgrid,siggrid = pickle.load(open(bigDataDir+"szgrid_"+expName+"_"+gridName+ "_v" + version+".pkl",'rb'))  

        self.cc = ClusterCosmology(self.fparams,self.constDict,clTTFixFile=self.clttfile)
        self.SZProp = SZ_Cluster_Model(self.cc,self.clusterDict,rms_noises = noise,fwhms=beam,freqs=freq,lknee=lknee,alpha=alpha)
        self.HMF = Halo_MF(self.cc,self.mgrid,self.zgrid)
        self.HMF.sigN = siggrid.copy()
        #self.dndm_SZ = self.HMF.dn_dmz_SZ(self.SZProp)

    def dVdz_fine(self,zarr):
        DA_z = self.HMF.cc.results.angular_diameter_distance(zarr)
        dV_dz = DA_z**2 * (1.+zarr)**2
        #NOT SURE we need this for loop
        for i in range (zarr.size):
            dV_dz[i] /= (self.HMF.cc.results.h_of_z(zarr[i]))
        dV_dz *= (old_div(self.HMF.cc.H0,100.))**3. # was h0
        return dV_dz

    def ntilde(self):
        dndm_SZ = self.HMF.dn_dmz_SZ(self.SZProp)
        ans = np.trapz(dndm_SZ,dx=np.diff(self.HMF.M200,axis=0),axis=0)
        return ans

    def ntilde_interpol(self,zarr_int):
        ntil = self.ntilde()
        z_arr = self.HMF.zarr

        #n = len(z_arr)
        #k = 4 # 5th degree spline 
        #s = 20.*(n - np.sqrt(2*n))* 1.3 # smoothing factor 

        #ntilde_spline = UnivariateSpline(z_arr, np.log(ntil), k=k, s=s)
        #ans = np.exp(ntilde_spline(zarr_int))
        f_int = interp1d(z_arr, np.log(ntil),kind='slinear')
        ans = np.exp(f_int(zarr_int)) 

        return ans

    def b_eff_z(self):
        ''' 
        effective linear bias wieghted by number density
        '''
        nbar = self.ntilde()

        z_arr = self.HMF.zarr
        dndm_SZ = self.HMF.dn_dmz_SZ(self.SZProp)
        
        R = tinker.radius_from_mass(self.HMF.M200,self.cc.rhoc0om)
        sig = np.sqrt(tinker.sigma_sq_integral(R, self.HMF.pk, self.HMF.kh))

        blin = tinker.tinker_bias(sig,200.)
        beff = old_div(np.trapz(dndm_SZ*blin,dx=np.diff(self.HMF.M200,axis=0),axis=0), nbar)

        return beff

    def non_linear_scale(self,z,M200): #?Who are you?

        zdiff = np.abs(self.HMF.zarr - z)
        use_z = np.where(zdiff == np.min(zdiff))[0]

        R = tinker.radius_from_mass(M200,self.cc.rhoc0om)
        
        sig = np.sqrt(tinker.sigma_sq_integral(R, self.HMF.pk[use_z,:], self.HMF.kh))

        #print sig[:,0],sig[0,:]
        print(sig.shape)
        print(self.HMF.kh.shape)
        sig1 = sig[0,:]
        print(sig1)
        sigdiff = np.abs(sig1 - 1.)
        use_sig = np.where(sigdiff == np.min(sigdiff))[0]
        print(use_sig)
        
        

        return old_div(1.,(R[use_sig])),sig1[use_sig],self.HMF.zarr[use_z]

# norm is off by 4 pi from ps_bar
    def Norm_Sfunc(self,fsky):
        #z_arr = self.HMF.zarr
        #Check this
        nbar = self.ntilde()
        ans = self.HMF.dVdz*nbar**2*np.diff(self.HMF.zarr_edges)
        return ans*4.*np.pi*fsky

    def fine_sfunc(self, fsky, nsubsamples):
        zs = self.HMF.zarr
        zgridedges = self.HMF.zarr_edges

        fine_zgrid = np.empty((zs.size, nsubsamples))
        for i in range(zs.size):
            fine_zgrid[i,:] = np.linspace(zgridedges[i], zgridedges[i+1], nsubsamples)

        fine_zgrid = fine_zgrid[1:-1]

        ntils = self.ntilde_interpol(fine_zgrid)
        dvdz = np.array([self.dVdz_fine(zs) for zs in fine_zgrid])

        dz = fine_zgrid[0,1] - fine_zgrid[0,0]

        assert np.allclose(dz * np.ones(tuple(np.subtract(fine_zgrid.shape, (0,1)))),  np.diff(fine_zgrid,axis=1), rtol=1e-3)

        integral = np.trapz(dvdz * ntils**2, dx=dz)
        integral *= 4 * np.pi * fsky

        return integral

    def v0(self, fsky, nsubsamples):
        zs = self.HMF.zarr
        zgridedges = self.HMF.zarr_edges

        fine_zgrid = np.empty((zs.size, nsubsamples))
        for i in range(zs.size):
            fine_zgrid[i,:] = np.linspace(zgridedges[i], zgridedges[i+1], nsubsamples)

        fine_zgrid = fine_zgrid[1:-1]
        dvdz = np.array([self.dVdz_fine(zs) for zs in fine_zgrid])
        dz = fine_zgrid[0,1] - fine_zgrid[0,0]

        assert np.allclose(dz * np.ones(tuple(np.subtract(fine_zgrid.shape, (0,1)))),  np.diff(fine_zgrid,axis=1), rtol=1e-3)
        
        integral = np.trapz(dvdz, dx=dz)
        integral *= 4 * np.pi * fsky 
        return integral

    def ps_tilde(self,mu):        
        beff_arr = self.b_eff_z()[..., np.newaxis]
        mu_arr = mu[..., np.newaxis]
        logGrowth = self.cc.fgrowth(self.HMF.zarr)

        prefac = (beff_arr.T + logGrowth*mu_arr**2)**2
        prefac = prefac[..., np.newaxis]

        pklin = self.HMF.pk # not actually pklin
        pklin = pklin.T
        pklin = pklin[..., np.newaxis]

        ans = np.multiply(prefac,pklin.T).T
        return ans

    def ps_tilde_interpol(self, zarr_int, mu):
        ps_tils = self.ps_tilde(mu)
        zs = self.HMF.zarr
        ks = self.HMF.kh

        n = zs.size
        k = 4 # 5th degree spline
        s = 20.*(n - np.sqrt(2*n))* 1.3 # smoothing factor

        ps_interp = interp1d(zs, ps_tils, axis=1, kind='cubic')
        return ps_interp(zarr_int)

    def ps_bar(self,mu,fsky):
        z_arr = self.HMF.zarr
        nbar = self.ntilde()
        prefac =  self.HMF.dVdz*nbar**2*np.diff(z_arr)[2]/self.Norm_Sfunc(fsky)
        prefac = prefac[..., np.newaxis]
        ans = np.multiply(prefac, self.ps_tilde(mu).T).T
        #for i in range(len(z_arr)): 
        #    ans[:,:,i] = self.HMF.dVdz[i]*nbar[i]**2*ps_tilde[:,:,i]*np.diff(z_arr[i])/self.Norm_Sfunc(fsky)[i]
        return ans

    def fine_ps_bar(self,mu,fsky, nsubsamples):
        zs = self.HMF.zarr
        ks = self.HMF.kh
        zgridedges = self.HMF.zarr_edges
        
        values = np.empty((ks.size, zs.size, mu.size))

        fine_zgrid = np.empty((zs.size, nsubsamples))
        for i in range(zs.size):
            fine_zgrid[i,:] = np.linspace(zgridedges[i], zgridedges[i+1], nsubsamples)

        fine_zgrid = fine_zgrid[1:-1]
        ntils = self.ntilde_interpol(fine_zgrid)
        dvdz = np.array([self.dVdz_fine(zs) for zs in fine_zgrid])
        prefac = dvdz * ntils**2
        prefac = prefac[..., np.newaxis]
        ps_tils = self.ps_tilde_interpol(fine_zgrid, mu)
        
        integrand = prefac * ps_tils
        dz = fine_zgrid[0,1] - fine_zgrid[0,0]

        assert np.allclose(dz * np.ones(tuple(np.subtract(fine_zgrid.shape, (0,1)))),  np.diff(fine_zgrid,axis=1), rtol=1e-3)

        integral = np.trapz(integrand, dx=dz, axis=2)
        s_norm = self.fine_sfunc(fsky, nsubsamples)[..., np.newaxis]
        values = (integral * 4 * np.pi * fsky)/s_norm
        return values

    def V_eff(self,mu,fsky,nsubsamples):
        V0 = self.v0(fsky, nsubsamples)
        V0 = np.reshape(V0, (V0.size,1))

        nbar = self.ntilde()[1:-1]
        nbar = np.reshape(nbar, (nbar.size,1))

        ps = self.fine_ps_bar(mu,fsky, nsubsamples)
        npfact = np.multiply(nbar, ps)
        frac = npfact/(1. + npfact)

        ans = np.multiply(frac**2,V0)
        return ans

