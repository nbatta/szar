import numpy as np
from sympy.functions import coth
import camb
from camb import model
import time
import cPickle as pickle

from tinker import dn_dlogM
from tinker import tinker_params
import tinker as tinker
from szar.foregrounds import fgNoises

from orphics.tools.io import Plotter
from orphics.theory.cosmology import Cosmology
import orphics.theory.cosmology as cosmo
from orphics.tools.stats import timeit
from scipy.interpolate import interp1d
from orphics.analysis.flatMaps import interpolateGrid

import szar._fast as fast

def getA(fparams,constDict,zrange,kmax=11.):
    cc = ClusterCosmology(fparams,constDict,skipCls=True)
    if zrange[0]>=1.e-8: zrange = np.insert(zrange,0,0)
    cc.pars.set_matter_power(redshifts=zrange, kmax=kmax)
    cc.pars.Transfer.high_precision = True
    cc.pars.NonLinear = camb.model.NonLinear_none
    cc.results = camb.get_results(cc.pars)
    s8s =  cc.results.get_sigma8()[::-1]
    As = s8s[1:]/s8s[0]
    return s8s[0],As


def rebinN(Nmzq,pzCutoff,zbin_edges):
    if pzCutoff>=zbin_edges[-2]: return zbin_edges,Nmzq
    orig = Nmzq.copy()
    indexUpTo = np.where(pzCutoff>=zbin_edges)[0][-1]
 
    rebinned = np.hstack((Nmzq[:,:indexUpTo,:],Nmzq[:,indexUpTo:,:].sum(axis=1).reshape((orig.shape[0],1,orig.shape[2]))))

    new_z_edges = np.append(zbin_edges[:indexUpTo+1],zbin_edges[-1])
    assert rebinned.shape[1]==(new_z_edges.size-1)
    assert new_z_edges.size<zbin_edges.size
    
    return new_z_edges,rebinned

def getTotN(Nmzq,mexp_edges,z_edges,q_edges,returnNz=False):
    """Get total number of clusters given N/DmDqDz
    and the corresponding log10(mass), z and q grid edges
    """

    Ndq = np.multiply(Nmzq,np.diff(q_edges).reshape((1,1,q_edges.size-1)))
    Ndm = np.multiply(Ndq,np.diff(10**mexp_edges).reshape((mexp_edges.size-1,1,1)))
    Ndz = np.multiply(Ndm,np.diff(z_edges).reshape((1,z_edges.size-1,1)))
    
    N = Ndz.sum()
    if returnNz:
        return N,Ndm.sum(axis=(0,2))
    else:
        return N


def getTotNM200(Nmzq,m200_edges_z,z_edges,q_edges,returnNz=False):
    """Get total number of clusters given N/DmDqDz
    and the corresponding log10(mass), z and q grid edges
    """

    Ndq = np.multiply(Nmzq,np.diff(q_edges).reshape((1,1,q_edges.size-1)))

    dm = m200_edges_z[1:,:]-m200_edges_z[:-1,:]
    Ndm = np.multiply(Ndq,dm.reshape((m200_edges_z.shape[0]-1,z_edges.size-1,1)))
    
    
    Ndz = np.multiply(Ndm,np.diff(z_edges).reshape((1,z_edges.size-1,1)))
    
    N = Ndz.sum()
    if returnNz:
        return N,Ndm.sum(axis=(0,2))
    else:
        return N


def getNmzq(Nmzq,mexp_edges,z_edges,q_edges):
    """Get number of clusters in each (m,z,q) bin 
    given dN/DmDqDz and the corresponding 
    log10(mass), z and q grid edges
    """
    Ndq = np.multiply(Nmzq,np.diff(q_edges).reshape((1,1,q_edges.size-1)))
    Ndz = np.multiply(Ndq,np.diff(z_edges).reshape((1,z_edges.size-1,1)))
    Ndm = np.multiply(Ndz,np.diff(10**mexp_edges).reshape((mexp_edges.size-1,1,1)))
    return Ndm
    



def haloBias(Mexp_edges,z_edges,rhoc0om,kh,pk):

    z_arr = (z_edges[1:]+z_edges[:-1])/2.
    M_edges = 10**Mexp_edges
    Masses = (M_edges[1:]+M_edges[:-1])/2.
   
    ac = 0.75
    pc = 0.3
    dc = 1.69

    M = np.outer(Masses,np.ones([len(z_arr)]))
    R = tinker.radius_from_mass(M, rhoc0om)
    sigsq = tinker.sigma_sq_integral(R, pk[:,:], kh)

    return z_arr,1. + (((ac*(dc**2.)/sigsq)-1.)/dc) + 2.*pc/(dc*(1.+(ac*dc*dc/sigsq)**pc))



def sampleVarianceOverNsquareOverBsquare(cc,kh,pk,z_edges,fsky,lmax=1000):
    zs = (z_edges[1:]+z_edges[:-1])/2.
    chis = cc.results.comoving_radial_distance(zs)

    chi_edges = cc.results.comoving_radial_distance(z_edges)
    dchis = np.diff(chi_edges)
    
    assert len(dchis)==len(chis)


    PofElls = []
    for i,chi in enumerate(chis):
        pfunc = interp1d(kh,pk[i,:])
        PofElls.append(lambda ell,chi: pfunc(ell*cc.h/chi))
    
    import healpy as hp
    frac = 1.-fsky
    nsides_allowed = 2**np.arange(5,13,1)
    nside_min = int((lmax+1.)/3.)
    nside = nsides_allowed[nsides_allowed>nside_min][0]
    npix = 12*(nside**2)

    hpmap = np.ones(npix)/(1.-frac)
    hpmap[:int(npix*frac)] = 0
    alms_original = hp.map2alm(hpmap)


    ellrange  = np.arange(2,lmax,1)

    powers = []
    for i,(chi,dchi) in enumerate(zip(chis,dchis)):
        Pl = PofElls[i](ellrange,chi)
        Pl = np.insert(Pl,0,0)
        Pl = np.insert(Pl,0,0)
        alms = hp.almxfl(alms_original.copy(),np.sqrt(Pl))
        power = (alms*alms.conjugate()).real/chi/chi/dchi
        powers.append(power.sum())
        # print i,powers[-1]

    powers = np.array(powers)

    return powers

def f_nu(constDict,nu):
    c = constDict
    mu = c['H_CGS']*(1e9*nu)/(c['K_CGS']*c['TCMB'])
    ans = mu*coth(mu/2.0) - 4.0
    return np.float(ans)


class ClusterCosmology(Cosmology):
    def __init__(self,paramDict=cosmo.defaultCosmology,constDict=cosmo.defaultConstants,lmax=None,clTTFixFile=None,skipCls=False,pickling=False):
        Cosmology.__init__(self,paramDict,constDict,lmax,clTTFixFile,skipCls,pickling)
        self.rhoc0om = self.rho_crit0H100*self.om
        
    def E_z(self,z):
        #hubble function
        ans = self.results.hubble_parameter(z)/self.paramDict['H0'] # 0.1% different from sqrt(om*(1+z)^3+ol)
        return ans

    def rhoc(self,z):
        #critical density as a function of z
        ans = self.rho_crit0H100*self.E_z(z)**2.
        return ans
    
        
    
    def rdel_c(self,M,z,delta):
        #spherical overdensity radius w.r.t. the critical density
        rhocz = self.rhoc(z)
        M = np.atleast_1d(M)
        return fast.rdel_c(M,z,delta,rhocz)

    def rdel_m(self,M,z,delta):
        #spherical overdensity radius w.r.t. the mean matter density
        return fast.rdel_m(M,z,delta,self.rhoc0om)

    def Mass_con_del_2_del_mean200(self,Mdel,delta,z):
        #Mass conversion critical to mean overdensity, needed because the Tinker Mass function uses mean matter
        rhocz = self.rhoc(z)
        ERRTOL = self.c['ERRTOL']        
        return fast.Mass_con_del_2_del_mean200(Mdel,delta,z,rhocz,self.rhoc0om,ERRTOL)

    def Mdel_to_cdel(self,M,z,delta):
        # Converts M to c where M is defined wrt delta overdensity relative to *critical* density at redshift of cluster.
        M200 = self.Mass_con_del_2_del_mean200(np.array(M).reshape((1,)),delta,z)[0]
        c200 = fast.con_M_rel_duffy200(M200, z)
        Rdelc = self.rdel_c(M,z,delta) # Rdel_crit in Mpc/h
        R200m = self.rdel_m(M200,z,200.) # R200_mean in Mpc/h
        c = c200*(Rdelc/R200m)**(1./3.)
        return c[0]


class Halo_MF:
    #@timeit
    def __init__(self,clusterCosmology,Mexp_edges,z_edges,kh=None,powerZK=None,kmin=1e-4,kmax=11.,knum=200):

        # update self.sigN (20 mins) and self.Pfunc if changing experiment
        # update self.cc or self.pk if changing cosmology
        # update self.Pfunc if changing scaling relation parameters


        self.cc = clusterCosmology

        zcenters = (z_edges[1:]+z_edges[:-1])/2.
        self.zarr_edges = z_edges
        self.zarr = zcenters
        if powerZK is None:
            self.kh, self.pk = self._pk(self.zarr,kmin,kmax,knum)
        else:
            assert kh is not None
            self.kh = kh
            self.pk = powerZK
            
        self.DAz = self.cc.results.angular_diameter_distance(self.zarr)        
        self._initdVdz(self.zarr)

        self.sigN = None
        self.YM = None
        self.Pfunc_qarr = None
        self.Pfunc = None

        M_edges = 10**Mexp_edges
        self.M_edges = M_edges
        M = (M_edges[1:]+M_edges[:-1])/2.
        Mexp = np.log10(M)
        #M = 10.**Mexp
        self.Mexp = Mexp
        self.M = M
        self.M200 = np.zeros((M.size,self.zarr.size))
        self.M200_edges = np.zeros((M_edges.size,self.zarr.size))
        self.zeroTemplate = self.M200.copy()

        for i in xrange(self.zarr.size):
            self.M200[:,i] = self.cc.Mass_con_del_2_del_mean200(M,500,self.zarr[i])

        for i in xrange(self.zarr.size):
            self.M200_edges[:,i] = self.cc.Mass_con_del_2_del_mean200(M_edges,500,self.zarr[i])
            

    def _pk(self,zarr,kmin,kmax,knum):
        self.cc.pars.set_matter_power(redshifts=zarr, kmax=kmax)
        self.cc.pars.Transfer.high_precision = True
        
        self.cc.pars.NonLinear = model.NonLinear_none
        self.cc.results = camb.get_results(self.cc.pars)
        kh, z, powerZK = self.cc.results.get_matter_power_spectrum(minkh=kmin, maxkh=kmax, npoints = knum)


        return kh, powerZK


    
    
    def _initdVdz(self,z_arr):
        #dV/dzdOmega
        DA_z = self.DAz 
        dV_dz = DA_z**2 * (1.+z_arr)**2
        for i in xrange (z_arr.size):
            dV_dz[i] /= (self.cc.results.h_of_z(z_arr[i]))
        dV_dz *= (self.cc.H0/100.)**3. # was h0
        self.dVdz = dV_dz
        #return dV_dz

    def dn_dM(self,M,delta):
        # dN/dmdV
        #Mass Function
        delts = self.zarr*0. + delta
        dn_dlnm = dn_dlogM(M,self.zarr,self.cc.rhoc0om,delts,self.kh,self.pk,'comoving')
        dn_dm = dn_dlnm/M
        return dn_dm
    
    def N_of_Mz(self,M,delta):
        #dN/dzdOmega
        
        dn_dm = self.dn_dM(M,delta)
        dV_dz = self.dVdz
        N_dzdm = dn_dm[:,:] * dV_dz[:]
        return N_dzdm

    def N_of_z(self):
        # dN/dz(z) = 4pi fsky \int dm dN/dzdmdOmega

        z_arr = self.zarr
        dn_dzdm = self.N_of_Mz(self.M200,200.)
        N_z = np.zeros(z_arr.size)
        for i in xrange(z_arr.size):
            N_z[i] = np.dot(dn_dzdm[:,i],np.diff(self.M200_edges[:,i]))

        return N_z*4.*np.pi

    def updateSigN(self,SZCluster,tmaxN=5,numts=1000):
        zs = self.zarr
        M = self.M

        print "Calculating variance grid. This is slow..."

        sigN = self.zeroTemplate.copy()

        for i in xrange(zs.size):
            for j in xrange(M.size):
                var = SZCluster.quickVar(M[j],zs[i],tmaxN,numts)
                sigN[j,i] = np.sqrt(var)
             
        self.sigN = sigN.copy()

    def updateYM(self,SZCluster):
        zs = self.zarr
        M = self.M

        YM = self.zeroTemplate
            

        for i in xrange(zs.size):
            for j in xrange(M.size):
                YM[j,i] = SZCluster.Y_M(M[j],zs[i])
        self.YM = YM


    def updatePfunc(self,SZCluster):
        print "updating"
        self.Pfunc = SZCluster.Pfunc(self.sigN.copy(),self.M.copy(),self.zarr.copy())

    def updatePfunc_qarr(self,SZCluster,q_arr):
        print "Calculating P_func_qarr. This takes a while..."
        self.Pfunc_qarr = SZCluster.Pfunc_qarr(self.sigN.copy(),self.M,self.zarr,q_arr)

    def N_of_z_SZ(self,SZCluster):
        # this is dN/dz(z) with selection

        z_arr = self.zarr
        
        if self.sigN is None: self.updateSigN(SZCluster)
        if self.Pfunc is None: self.updatePfunc(SZCluster)
        P_func = self.Pfunc.copy()

        print self.sigN
        print self.Pfunc
        sys.exit()

        dn_dzdm = self.dn_dM(self.M200,200.)
        N_z = np.zeros(z_arr.size)
        for i in xrange (z_arr.size):
            N_z[i] = np.dot(dn_dzdm[:,i]*P_func[:,i],np.diff(self.M200_edges[:,i]))

        print P_func
        sys.exit()
        return N_z* self.dVdz[:]*4.*np.pi

    def N_of_mz_SZ(self,SZCluster):

        z_arr = self.zarr
        
        if self.sigN is None: self.updateSigN(SZCluster)
        if self.Pfunc is None: self.updatePfunc(SZCluster)
        P_func = self.Pfunc

        dn_dzdm = self.dn_dM(self.M200,200.)
        N_z = np.zeros(z_arr.size)
        N_mz = np.multiply(dn_dzdm[:,:]*P_func[:,:],np.diff(self.M200_edges[:,:],axis=0))


        return N_mz* self.dVdz[:]*4.*np.pi

    def Mass_err (self,fsky,mass_err,SZCluster):
        alpha_ym = self.cc.paramDict['alpha_ym'] 
        z_arr = self.zarr

        if self.sigN is None: self.updateSigN(SZCluster)
        if self.YM is None: self.updateYM(SZCluster)
        if self.Pfunc is None: self.updatePfunc(SZCluster)
        P_func = self.Pfunc

        dn_dVdm = self.dn_dM(self.M200,200.)
        dV_dz = self.dVdz

        N_z = np.zeros(z_arr.size)
        N_tot_z = np.zeros(z_arr.size)
        for i in xrange(z_arr.size):
            N_z[i] = np.dot(dn_dVdm[:,i]*P_func[:,i] / (mass_err[:,i]**2 + alpha_ym**2 * (self.sigN[:,i]/self.YM[:,i])**2),np.diff(self.M200_edges[:,i]))
            N_tot_z[i] = np.dot(dn_dVdm[:,i]*P_func[:,i],np.diff(self.M200_edges[:,i]))
        err_WL_mass = 4.*np.pi* fsky*np.dot(N_z*dV_dz[:],np.diff(self.zarr_edges))
        Ntot = 4.*np.pi* fsky*np.dot(N_tot_z*dV_dz[:],np.diff(self.zarr_edges))

        return 1./err_WL_mass,Ntot

    def N_of_mqz_SZ (self,mass_err,q_edges,SZCluster):
        # this is 3D grid for fisher matrix

        q_arr = (q_edges[1:]+q_edges[:-1])/2.

        z_arr = self.zarr
        M_arr =  np.outer(self.M,np.ones([len(z_arr)]))
        dNdzmq = np.zeros([len(self.M),len(z_arr),len(q_arr)])

        m_wl = self.Mexp
        
        if self.sigN is None: self.updateSigN(SZCluster)
        if self.Pfunc_qarr is None: self.updatePfunc_qarr(SZCluster,q_arr)
        P_func = self.Pfunc_qarr

        dn_dVdm = self.dn_dM(self.M200,200.)
        dV_dz = self.dVdz

        # N = M_arr.copy()*0.
        # print "Checking norm of mass calibration..."
        # mexp_int = m_wl
        # m_int_edges = self.M_edges
        # #mass_err *= 1.e-1

        # # mexp_int_edges = np.arange(9.0,16.0,0.01)
        # # m_int_edges = 10**mexp_int_edges
        # # m_int = (m_int_edges[1:]+m_int_edges[:-1])/2.
        # # mexp_int = np.log10(m_int)
        # for i in xrange (z_arr.size):
        #     for j in xrange (self.Mexp.size):
        #         N[j,i] = np.dot(SZCluster.Mwl_prob(10**(mexp_int),M_arr[j,i],mass_err[j,i]),np.diff(m_int_edges))
        # from orphics.tools.io import Plotter
        # import os
        # mmin = self.Mexp.min()
        # mmax = self.Mexp.max()
        # zmin = self.zarr.min()
        # zmax = self.zarr.max()
        # pgrid = np.rot90(N)
        # pl = Plotter(labelX="$\\mathrm{log}_{10}(M)$",labelY="$z$",ftsize=14)
        # pl.plot2d(pgrid,extent=[mmin,mmax,zmin,zmax],labsize=14,aspect="auto")
        # pl.done(os.environ['WWW']+"normMassCalib.png")

        # pgrid = mass_err
        # pl = Plotter(labelX="$\\mathrm{log}_{10}(M)$",labelY="$z$",ftsize=14)
        # pl.plot2d(pgrid,extent=[mmin,mmax,zmin,zmax],labsize=14,aspect="auto")
        # pl.done(os.environ['WWW']+"massgrid.png")

        # sys.exit()
        
        # \int dm  dn/dzdm
        for kk in xrange(q_arr.size):
            for jj in xrange(m_wl.size):
                for i in xrange (z_arr.size):
                    dM = np.diff(self.M200_edges[:,i])
                    dNdzmq[jj,i,kk] = np.dot(dn_dVdm[:,i]*P_func[:,i,kk]*SZCluster.Mwl_prob(10**(m_wl[jj]),M_arr[:,i],mass_err[:,i]),dM) * dV_dz[i]*4.*np.pi
        
        return dNdzmq

    def Cl_ell(self,ell,SZCluster):
        #fix the mass and redshift ranges 
        M = self.M#10**np.arange(11.0, 16, .1)
        #dM = np.gradient(M)
        
        #zmax = 6.0
        #zmin = 0.01
        z_arr = self.zarr
        dz = np.gradient(z_arr)
        #delz = (zmax-zmin)/100. # 140 redshift restriction
        #zbin_temp = np.arange(zmin,zmax,delz)
        #z_arr = np.insert(zbin_temp,0,0.0)
        #dz = np.diff(z_arr)
        #ell = np.arange(1000,3000,1000)
        
        M200 = np.outer(M,np.zeros([len(z_arr)]))
        dM200 = np.outer(M,np.zeros([len(z_arr)]))
        formfac = np.zeros((len(ell),len(M),len(z_arr)))
        
#        np.outer(ell,np.outer(M,np.zeros([len(z_arr)])))
        #print formfac.shape
        
        for i in xrange (z_arr.size):
            M200[:,i] = self.cc.Mass_con_del_2_del_mean200(M/(self.cc.H0/100.),500,z_arr[i])
            dM200[:,i] = np.gradient(M200[:,i])
            if (i > 0): 
                for j in xrange (len(M)):
                    for k in xrange (len(ell)):
                        formfac[k,j,i] = SZCluster.Prof_tilde(ell[k],M[j]/(self.cc.H0/100.),z_arr[i])
        
        dn_dm = self.dn_dM(M200,200.)
        dV_dz = self.dVdz
        
        ans = np.zeros(len(ell))
        for k in xrange (len(ell)):
            for i in xrange (z_arr.size):
                ans[k] += 4*np.pi *dV_dz[i] * dz[i] * np.trapz( dn_dm[:,i] * formfac[k,:,i]**2,M200[:,i])
        #print dn_dm.shape, formfac.shape
        #ans  =1.    
        return ans

    
