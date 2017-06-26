import numpy as np
from scipy.special import j0
import camb
from camb import model
from szar.counts import Halo_MF

class pairwise:
    def __init__(self,clusterCosmology,Mexp_edges,z_edges,kh=None,powerZK=None):

        self.cc = clusterCosmology

        self.HMF = Halo_MF(self.cc,Mexp_edges,z_edges)

        if powerZK is None:
            self.kh_lin, self.pk_lin = self._pk_lin(self.zarr,kmin,kmax,knum)
        else:
            assert kh is not None
            self.kh_lin = kh
            self.pk_lin = powerZK

        self.HMF.dn_dM(self.HMF.M200,200.)

        def _pk_lin(self,zarr,kmin,kmax,knum): #Linear PK
            self.cc.pars.set_matter_power(redshifts=zarr, kmax=kmax)
            self.cc.pars.Transfer.high_precision = True

            self.cc.pars.Linear = model.Linear_none
            self.cc.results = camb.get_results(self.cc.pars)
        
            kh, z, powerZK = self.cc.results.get_matter_power_spectrum(minkh=kmin, maxkh=kmax, npoints = knum)

            return kh, powerZK

        def halo_bias(self,)

        self.zeta = 
    
        self.zeta_bar =

    def meanvel():
        
