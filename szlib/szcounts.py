import numpy as np
from scipy import special
from sympy.functions import coth
import camb
from camb import model
import time
import cPickle as pickle

from Tinker_MF import dn_dlogM
from Tinker_MF import tinker_params
import Tinker_MF as tinker

from orphics.tools.output import Plotter
from orphics.theory.cosmology import Cosmology
from orphics.tools.stats import timeit
from scipy.interpolate import interp1d
from orphics.analysis.flatMaps import interpolateGrid

import szlib.szlibNumbafied as fast
from scipy.special import j0

def gaussian(xx, mu, sig):
    return 1./(sig * np.sqrt(2*np.pi)) * np.exp(-1.*(xx - mu)**2 / (2. * sig**2.))

def haloBias(Mexp,rhoc0m,kh,pk):
    ac = 0.75
    pc = 0.3
    dc = 1.69

    R = tinker.radius_from_mass(10**Mexp, rhoc0om)
    sigsq = tinker.sigma_sq_integral(R, pk[:,:], kh)


    return 1. + ((ac*dc**2./sigsq-1.)/dc) + 2.*pc/(dc*(1.+(ac*dc*dc/sigsq)**pc))

def sampleVarianceOverNsquare(cc,Mexprange,z_arr,lmax=1000):
    hmf = Halo_MF(cc)
    self.cc = cc
    self.kh, self.pk = hmf.kh, hmf.pk #pk(z_arr)
    self.Mexp = Mexprange
    self.chis = self.cc.results.comoving_radial_distance(z_arr)
    self.zs = z_arr


    self.PofElls = []
    for i,chi in enumerate(self.chis):
        pfunc = interp1d(self.kh,self.pk[i,:])
        self.PofElls.append(lambda ell,chi: pfunc(ell*self.cc.h/chi))
    

class SampleVariance(object):
    @timeit
    def __init__(self,cc,Mexprange,z_arr,lmax=1000):
        hmf = Halo_MF(cc)
        self.cc = cc
        self.kh, self.pk = hmf.kh, hmf.pk #hmf.pk(z_arr)
        self.Mexp = Mexprange
        self.chis = self.cc.results.comoving_radial_distance(z_arr)
        self.zs = z_arr

        
        self.PofElls = []
        for i,chi in enumerate(self.chis):
            pfunc = interp1d(self.kh,self.pk[i,:])
            self.PofElls.append(lambda ell,chi: pfunc(ell*self.cc.h/chi))
        

    def haloBias(self):
        ac = 0.75
        pc = 0.3
        dc = 1.69

        R = tinker.radius_from_mass(10**self.Mexp, self.cc.rhoc0om)
        R = np.resize(R,(self.Mexp.size,self.pk[:,0].size))
        sigsq = tinker.sigma_sq_integral(R, self.pk[:,:], self.kh)


        return 1. + ((ac*dc**2./sigsq-1.)/dc) + 2.*pc/(dc*(1.+(ac*dc*dc/sigsq)**pc))

    @timeit
    def sample_variance_overNsquared(self,fsky):
        import healpy as hp
        frac = 1.-fsky
        nsides_allowed = 2**np.arange(5,13,1)
        nside_min = int((lmax+1.)/3.)
        nside = nsides_allowed[nsides_allowed>nside_min][0]
        npix = 12*nside**2

        hpmap = np.ones(npix)/(1.-frac)
        hpmap[:int(npix*frac)] = 0
        alms_original = hp.map2alm(hpmap)

        
        ellrange  = np.arange(2,lmax,1)
        dchis = np.diff(self.chis)
        
        for i,(chi,dchi) in enumerate(zip(self.chis,dchis)):
            Pl = self.PofElls[i](ellrange,chi)
            Pl = np.insert(Pl,0,0)
            Pl = np.insert(Pl,0,0)
            alms = hp.almxfl(alms_original.copy(),np.sqrt(Pl))
            power = (alms*alms.conjugate()).real/chi/chi/dchi
            print power.sum()



def f_nu(cc,nu):
    mu = cc.c['H_CGS']*(1e9*nu)/(cc.c['K_CGS']*cc.c['TCMB'])
    ans = mu*coth(mu/2.0) - 4.0
    return np.float(ans)


class ClusterCosmology(Cosmology):
    def __init__(self,paramDict,constDict,lmax=None,clTTFixFile=None,skipCls=False,pickling=False):
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

class Halo_MF:

    @timeit
    def __init__(self,clusterCosmology,Mexp,zcenters,kh=None,powerZK=None,kmin=2e-5,kmax=11.,knum=200):
        self.cc = clusterCosmology

        zcenters = np.insert(zcenters,0,0.0)

        self.zarr = zcenters
        if powerZK is None:
            self.kh, self.pk = self._pk(self.zarr,kmin,kmax,knum)
        self.DAz = self.cc.results.angular_diameter_distance(self.zarr)        
        self._initdVdz(self.zarr)

        M = 10.**Mexp
        self.M = M
        self.M200 = np.outer(M,np.zeros([len(self.zarr)]))

        for ii in xrange(len(self.zarr)-1):
            i = ii + 1
            self.M200[:,i] = self.cc.Mass_con_del_2_del_mean200(M,500,self.zarr[i])

        # from orphics.tools.io import Plotter
        # pl = Plotter()
        # pl.plot2d(self.M200)
        # pl.done("output/debug.png")

    def _pk(self,zarr,kmin=2e-5,kmax=11.,knum=200):
        self.cc.pars.set_matter_power(redshifts=zarr, kmax=kmax)
        self.cc.pars.Transfer.high_precision = True
        
        self.cc.pars.NonLinear = model.NonLinear_none
        self.cc.results = camb.get_results(self.cc.pars)
        kh, z, powerZK = self.cc.results.get_matter_power_spectrum(minkh=kmin, maxkh=kmax, npoints = knum)
        #kh, z, pk = self.cc.results.get_matter_power_spectrum(minkh=1e-4, maxkh=11, npoints = 200)
        return kh, powerZK


    
    
    def _initdVdz(self,z_arr):
        #dV/dzdOmega
        DA_z = self.DAz 
        dV_dz = DA_z**2 * (1.+z_arr)**2
        for i in xrange (len(z_arr)):
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
        N_dzdm = dn_dm[:,1:] * dV_dz[1:]
        return N_dzdm

    def N_of_z(self,fsky):
        # dN/dz(z) = 4pi fsky \int dm dN/dzdmdOmega
        

        z_arr = self.zarr
            
        dn_dzdm = self.N_of_Mz(self.M200,200.)

        N_z = np.zeros(len(z_arr) - 1)
        for i in xrange (len(z_arr) - 1):
            N_z[i] = np.trapz(dn_dzdm[:,i],self.M200[:,i+1],np.diff(self.M200[:,i+1]))

        return N_z*4.*np.pi*fsky



    def N_of_z_SZ(self,fsky,SZCluster,tmaxN=5,numts=1000):
        # this is dN/dz(z) with selection

        z_arr = self.zarr

        P_func = SZCluster.Pfunc(self.M,self.M200,self.zarr,tmaxN,numts)

        dn_dzdm = self.dn_dM(self.M200,200.)
        N_z = np.zeros(len(z_arr) - 1)
        for i in xrange (len(z_arr) - 1):
            N_z[i] = np.trapz(dn_dzdm[:,i+1]*P_func[:,i+1],self.M200[:,i+1],np.diff(self.M200[:,i+1]))


        return N_z* self.dVdz[1:]*4.*np.pi*fsky


    def Mass_err (self,fsky,mass_err,SZCluster,tmaxN=5,numts=1000):
        alpha_ym = self.cc.paramDict['alpha_ym'] 
        z_arr = self.zarr

        P_func,YM = SZCluster.Pfunc(self.M,self.M200,self.zarr,tmaxN,numts,doYM=True)

        dn_dVdm = self.dn_dM(self.M200,self.zarr,200.)
        dV_dz = self.dVdz

        N_z = np.zeros(len(z_arr) - 1)
        N_tot_z = np.zeros(len(z_arr) - 1)
        for i in xrange (len(z_arr) - 1):
            N_z[i] = np.trapz(dn_dVdm[:,i+1]*P_func[:,i+1] / (mass_err[:,i]**2 + alpha_ym**2 * (sigN[:,i+1]/YM[:,i+1])**2),self.M200[:,i+1])
            N_tot_z[i] = np.trapz(dn_dVdm[:,i+1]*P_func[:,i+1],self.M200[:,i+1])
        err_WL_mass = 4.*np.pi* fsky*np.trapz(N_z*dV_dz[1:],self.zarr)
        Ntot = 4.*np.pi* fsky*np.trapz(N_tot_z*dV_dz[1:],self.zarr)

        return 1./err_WL_mass,Ntot

    def N_of_mqz_SZ (self,mass_err,q_arr,beams,noises,freqs,clusterDict,lknee,alpha,tmaxN=5,numts=1000):
        # this is 3D grid for fisher matrix 

        z_arr = self.zarr
        M_arr =  np.outer(M,np.ones([len(z_arr)]))
        dNdzmq = np.zeros([len(self.M),len(z_arr)-1,len(q_arr)])

        
        P_func,YM = SZCluster.Pfunc(self.M,self.M200,self.zarr,tmaxN,numts,doYM=True)

        dn_dVdm = self.dn_dM(M200,z_arr,200.)
        dV_dz = self.dVdz

        # \int dm  dn/dzdm 
        for kk in xrange(len(q_arr)):
            for jj in xrange(len(m_wl)):
                for i in xrange (len(z_arr) - 1):
                    dM = np.diff(M200[:,i+1])
                    dNdzmq[jj,i,kk] = np.trapz(dn_dVdm[:,i+1]*P_func[:,i,kk]*SZProf.Mwl_prob(10**(m_wl[jj]),M_arr[:,i+1],mass_err[:,i]),M200[:,i+1],dM) * dV_dz[i+1]*4.*np.pi
                   
        return dNdzmq

class SZ_Cluster_Model:
    def __init__(self,clusterCosmology,clusterDict,fwhms=[1.5],rms_noises =[1.], freqs = [150.],lmax=8000,lknee=0.,alpha=1.,dell=10,pmaxN=5,numps=1000,nMax=1):
        self.cc = clusterCosmology
        self.P0 = clusterDict['P0']
        self.xc = clusterDict['xc']
        self.al = clusterDict['al']
        self.gm = clusterDict['gm']
        self.bt = clusterDict['bt']

        self.scaling = self.cc.paramDict
        
        # build noise object
        self.dell = 10
        self.nlinv = 0.
        self.nlinv2 = 0.
        self.evalells = np.arange(2,lmax,self.dell)
        for freq,fwhm,noise in zip(freqs,fwhms,rms_noises):
            freq_fac = (f_nu(self.cc,freq))**2


            nells = self.cc.clttfunc(self.evalells)+( self.noise_func(self.evalells,fwhm,noise,lknee,alpha) / self.cc.c['TCMBmuK']**2.)
            self.nlinv2 += (freq_fac)/nells

            nells += (self.rad_ps(self.evalells,freq,freq) + self.cib_p(self.evalells,freq,freq) + \
                      self.cib_c(self.evalells,freq,freq) + self.ksz_temp(self.evalells)) \
                      / self.cc.c['TCMBmuK']**2. / ((self.evalells+1.)*self.evalells) * 2.* np.pi 

            self.nlinv += (freq_fac)/nells
        self.nl = 1./self.nlinv
        self.nl2 = 1./self.nlinv2  

        # ls = self.evalells
        # pl = Plotter(scaleY='log')
        # pl.add(ls,self.cc.theory.lCl('TT',self.evalells)*ls**2.)
        # pl.add(ls,self.nl*ls**2.,label="no freq")
        # pl.add(ls,self.nl*freq_fac*ls**2.,label="freq")
        # pl.legendOn(loc="upper right")
        # pl.done("output/cmb.png")
        # sys.exit()


           
        # build profile for quickvar
        # p(x)
        c = self.xc
        alpha = self.al
        beta = self.bt
        gamma = self.gm
        p = lambda x: 1./(((c*x)**gamma)*((1.+((c*x)**alpha))**((beta-gamma)/alpha)))


        # g(x) = \int dz p(sqrt(z**2+x**2))
        pmaxN = pmaxN
        numps = numps
        pzrange = np.linspace(-pmaxN,pmaxN,numps)
        self.g = lambda x: np.trapz(p(np.sqrt(pzrange**2.+x**2.)),pzrange,np.diff(pzrange))
        #print "g(0) ", self.g(0)
        self.gxrange = np.linspace(0.,nMax,numps)        
        self.gint = np.array([self.g(x) for x in self.gxrange])

        # pl = Plotter()
        # pl.add(gxrange,gint)
        # pl.done("output/gint.png")

        

        # old code starts here

        self.NNR = 1000
        self.drint = 1e-4
        self.rad = (np.arange(1.e5) + 1.0)*self.drint #in MPC (max rad is 10 MPC)
        self.rad2 = self.rad**2.




        #self.dtht = 0.0000025
        self.dtht = 0.00001



    #@timeit
    def quickVar(self,M,z,tmaxN=5.,numts=1000):




        # R500 in MPc, DAz in MPc, th500 in radians
        R500 = self.cc.rdel_c(M,z,500.).flatten()[0] # R500 in Mpc/h
        #print R500
        DAz = self.cc.results.angular_diameter_distance(z) * (self.cc.H0/100.) 
        th500 = R500/DAz
        #print "t500", th500 
        # gnorm = 2pi th500^2  \int dx x g(x)
        gnorm = 2.*np.pi*(th500**2.)*np.trapz(self.gxrange*self.gint,self.gxrange,np.diff(self.gxrange))

        # u(th) = g(th/th500)/gnorm
        u = lambda th: self.g(th/th500)/gnorm
        thetamax = tmaxN * th500
        thetas = np.linspace(0.,thetamax,numts)
        uint = np.array([u(t) for t in thetas])

        # \int dtheta theta j0(ell*theta) u(theta)
        ells = self.evalells
        integrand = lambda l: np.trapz(j0(l*thetas)*uint*thetas,thetas,np.diff(thetas))
        integrands = np.array([integrand(ell) for ell in ells])



        # varinv = \int dell 2pi ell integrand^2 / nl
        varinv = np.trapz((integrands**2.)*ells*2.*np.pi/self.nl,ells,np.diff(ells))
        var = 1./varinv

        return var

    def GNFW(self,xx):
        ans = self.P0 / ((xx*self.xc)**self.gm * (1 + (xx*self.xc)**self.al)**((self.bt-self.gm)/self.al))
        return ans

    @timeit
    def Pfunc(self,M,M200,z_arr,tmaxN=5,numts=1000,doYM=False):

        lnYmin = np.log(1e-14)
        dlnY = 0.1
        lnY = np.arange(lnYmin,lnYmin+13.,dlnY)

        # lnYmin = np.log(1e-13)
        # dlnY = 0.1
        # lnY = np.arange(lnYmin,lnYmin+10.,dlnY)
    
        rho_crit0m = self.cc.rhoc0om
        hh = self.cc.H0/100

        P_func = np.outer(M,np.zeros([len(z_arr)]))
        sigN = np.outer(M,np.zeros([len(z_arr)]))
        M_arr =  np.outer(M,np.ones([len(z_arr)]))
        if doYM: YM   =  np.outer(M,np.ones([len(z_arr)]))


        DA_z = self.cc.results.angular_diameter_distance(z_arr) * hh


        for ii in xrange (len(z_arr)-1):
            i = ii + 1
            for j in xrange(len(M)):
                var = self.quickVar(M[j],z_arr[i],tmaxN,numts)
                sigN[j,i] = np.sqrt(var)
                if doYM: YM[j,i] = SZProf.Y_M(M[j],z_arr[i])
                
            P_func[:,i] = self.P_of_q (lnY,M_arr[:,i],z_arr[i],sigN[:,i])#*dlnY

        if doYM: 
            return P_func,YM
        else:
            return P_func
    
    def Prof(self,r,M,z,R500):
        R500 = R500
        xx = r / R500
        M_fac = M / (3e14) * (100./70.)
        P500 = 1.65e-3 * (100./70.)**2 * M_fac**(2./3.) * self.cc.E_z(z) #keV cm^3
        ans = P500 * self.GNFW(xx)
        return ans

    def Prof_tilde(self,ell,M,z):
        dr = 0.01
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=cp.h0, ombh2=cp.ob*(cp.h0/100.)**2, omch2=(cp.om - cp.ob)*(cp.h0/100.)**2,)    
        results = camb.get_background(pars)
        DA_z = results.angular_diameter_distance(z) #* (70./100.)
        R500 = HP.rdel_c(M,z,500)
        rr = np.arange(dr,R500*5.0,dr)
        intgrl = np.sum(self.Prof(rr,M,z)*rr**2*np.sin(ell*rr/DA_z) / (ell*rr/DA_z) ) * dr
        ans = 4.0*np.pi/DA_z**2 * intgrl
        ans *= c.SIGMA_T/(c.ME*c.C**2)*c.MPC2CM*c.eV_2_erg*1000.0
        return ans
    
    def y2D_norm(self,M,z,tht,R500):

        r = np.arange(0.0001,100.,0.0001)
        prof = self.Prof(r,M,z,R500)
        profunc = interp1d(r,prof,bounds_error=True)        


        thtr5002 = (tht*R500)**2.
        P2D = tht * 0.0
        for ii in xrange(len(tht)):
            rint = np.sqrt(self.rad2 + thtr5002[ii])
            P2D[ii] = np.sum(profunc(rint))
            
        P2D *= 2.*self.drint
        P2D /= P2D[0]
        return P2D
    

    def y2D_tilde_norm(self,M,z,ell,thtc,thta,R500):
        ans = ell*0.
        y2D_use = self.y2D_norm(M,z,thta/thtc,R500)
        for ii in xrange(len(ell)):
            ans[ii] = np.sum(thta*special.jv(0,ell[ii]*thta)*y2D_use)*self.dtht
        return ans, y2D_use

    def noise_func(self,ell,fwhm,rms_noise,lknee=0.,alpha=0.):
        if lknee>1.e-3:
            atmFactor = (lknee/ell)**(-alpha)
        else:
            atmFactor = 0.
        rms = rms_noise * (1./60.)*(np.pi/180.)
        tht_fwhm = np.deg2rad(fwhm / 60.)
        ans = (atmFactor+1.) * (rms**2.) * np.exp((tht_fwhm**2.)*(ell**2.) / (8.*np.log(2.))) ## Add Hasselfield noise knee
        return ans

    def g_nu(self,nu):
        beta = (nu*1e9) * self.cc.c['H_CGS'] / (self.cc.c['K_CGS']*self.cc.c['TCMB'])
        ans = 2.* self.cc.c['H_CGS']**2 * (nu*1e9)**4 / (self.cc.c['C']**2 *self.cc.c['K_CGS'] * self.cc.c['TCMB']**2) \
            * np.exp(beta) * 1. / (np.exp(beta) - 1.)**2
        return ans

    def B_nu(self,Td,nu):
        beta = (nu*1e9) * self.cc.c['H_CGS'] / (self.cc.c['K_CGS']*Td)
        ans = 2.* self.cc.c['H_CGS'] * nu**3 / (self.cc.c['C'])**2 / (np.exp(beta) - 1.)
        return ans

    def rad_ps(self,ell,nu1,nu2):
        ans = self.cc.c['A_ps'] * (ell/self.cc.c['ell0sec']) ** 2 * (nu1*nu2/self.cc.c['nu0']**2) ** self.cc.c['al_ps'] \
            * self.g_nu(nu1) * self.g_nu(nu2) / (self.g_nu(self.cc.c['nu0']))**2
        return ans 
    
    def res_gal(self,ell,nu1,nu2):
        ans = cc.c['A_g'] * (ell/cc.c['ell0sec']) ** cc.c['n_g'] * (nu1*nu2/cc.c['nu0']**2) ** cc.c['al_g'] \
            * self.g_nu(nu1) * self.g_nu(nu2) / (self.g_nu(cc.c['nu0']))**2
        return ans 
    
    def cib_p(self,ell,nu1,nu2):

        mu1 = nu1**self.cc.c['al_cib']*self.B_nu(self.cc.c['Td'],nu1) * self.g_nu(nu1)
        mu2 = nu2**self.cc.c['al_cib']*self.B_nu(self.cc.c['Td'],nu2) * self.g_nu(nu2)
        mu0 = self.cc.c['nu0']**self.cc.c['al_cib']*self.B_nu(self.cc.c['Td'],self.cc.c['nu0']) \
            * self.g_nu(self.cc.c['nu0'])

        ans = self.cc.c['A_cibp'] * (ell/self.cc.c['ell0sec']) ** 2.0 * mu1 * mu2 / mu0**2
        return ans

    def cib_c(self,ell,nu1,nu2):
        mu1 = nu1**self.cc.c['al_cib']*self.B_nu(self.cc.c['Td'],nu1) * self.g_nu(nu1)
        mu2 = nu2**self.cc.c['al_cib']*self.B_nu(self.cc.c['Td'],nu2) * self.g_nu(nu2)
        mu0 = self.cc.c['nu0']**self.cc.c['al_cib']*self.B_nu(self.cc.c['Td'],self.cc.c['nu0']) \
            * self.g_nu(self.cc.c['nu0'])
 
        ans = self.cc.c['A_cibc'] * (ell/self.cc.c['ell0sec']) ** (2.-self.cc.c['n_cib']) * mu1 * mu2 / mu0**2
        return ans

    def ksz_temp(self,ell):
        el,ksz = np.loadtxt('input/ksz_BBPS.txt',unpack=True)
        elp,kszp = np.loadtxt('input/ksz_p_BBPS.txt',unpack=True)
        ans = np.interp(ell,el,ksz) * (1.65/1.5) + np.interp(ell,elp,kszp)
        return ans

    @timeit
    def filter_variance(self,M,z):


        R500 = self.cc.rdel_c(M,z,500.).flatten()[0]
        DAz = self.cc.results.angular_diameter_distance(z) * (self.cc.H0/100.)

        
        thtc = R500/DAz
        thta = np.arange(self.dtht,5.*thtc,self.dtht)  ### Changed 25 to 5 and it didn't change much
        ytilde, y2D_use = self.y2D_tilde_norm(M,z,self.evalells,thtc,thta,R500)
        y2dtilde_2 = (ytilde)**2
        var = np.sum(self.evalells*y2dtilde_2/self.nl)*self.dell#*self.freq_fac

        prof_int = 2.*np.pi*(np.sum((y2D_use*thta)[thta < 1.*thtc])*self.dtht)**2
        
        return prof_int/var
    
    
    def Y_M(self,MM,zz):
        DA_z = self.cc.results.angular_diameter_distance(zz) * (self.cc.H0/100.)
        
        Y_star = self.scaling['Y_star'] #= 2.42e-10 #sterads
        #dropped h70 factor
        alpha_ym = self.scaling['alpha_ym'] #1.79
        b_ym = self.scaling['b_ym'] #0.8
        beta_ym = self.scaling['beta_ym'] #= 0.66
        gamma_ym = self.scaling['gamma_ym']        
        beta_fac = np.exp(beta_ym*(np.log(MM/1e14))**2)
        #print beta_fac
        ans = Y_star*((b_ym)*MM/ 1e14)**alpha_ym *(DA_z/100.)**(-2.) * beta_fac * self.cc.E_z(zz) ** (2./3.) * (1. + zz)**gamma_ym
        #print (0.01/DA_z)**2
        return ans

    
    def Y_erf(self,Y_true,sigma_N):
        q = 6.
        sigma_Na = np.outer(sigma_N,np.ones(len(Y_true[0,:])))
        
        ans = 0.5 * (1. + special.erf((Y_true - q*sigma_Na)/(np.sqrt(2.)*sigma_Na)))
        return ans
    
    def P_of_Y (self,lnY,MM,zz):
        #Ysig = self.scaling['Ysig'] #0.127 


        Y = np.exp(lnY)
        Ma = np.outer(MM,np.ones(len(Y[0,:])))
        Ysig = self.scaling['Ysig'] * (1. + zz)**self.scaling['gammaYsig'] * (Ma/(self.cc.H0/100.)/6e14)**self.scaling['betaYsig']
        numer = -1.*(np.log(Y/self.Y_M(Ma,zz)))**2
        ans = 1./(Ysig * np.sqrt(2*np.pi)) * np.exp(numer/(2.*Ysig**2))
        return ans
    
    def P_of_q (self,lnY,MM,zz,sigma_N):
        lnYa = np.outer(np.ones(len(MM)),lnY)
        
        #DA_z = self.cc.results.angular_diameter_distance(zz) * (self.cc.H0/100.)
        
        sig_thresh = self.Y_erf(np.exp(lnYa),sigma_N)
        P_Y = self.P_of_Y(lnYa,MM, zz)

        ans = MM*0.0
        for ii in xrange(len(MM)):
            ans[ii] = np.trapz(P_Y[ii,:]*sig_thresh[ii,:],lnY,np.diff(lnY))
        return ans

    def P_of_qn (self,lnY,MM,zz,sigma_N,qarr):
        lnYa = np.outer(np.ones(len(MM)),lnY)
        #dY = np.outer(np.ones(len(MM)),np.gradient(np.exp(lnY)))
        
        #DA_z = self.cc.results.angular_diameter_distance(zz) * (self.cc.H0/100.)
        
        sig_thresh = self.q_prob(qarr,lnYa,sigma_N)
        P_Y = self.P_of_Y(lnYa,MM, zz)
        ans = MM*0.0
        for ii in xrange(len(MM)):
            ans[ii] = np.trapz(P_Y[ii,:]*sig_thresh[ii,:],lnY,np.diff(lnY))
        return ans


    def q_prob (self,q_arr,lnY,sigma_N):
        #Gaussian error probablity for SZ S/N
        sigma_Na = np.outer(sigma_N,np.ones(len(lnY[0,:])))
        Y = np.exp(lnY)
        ans = gaussian(q_arr,Y/sigma_Na,1.)        
        return ans

    def Mwl_prob (self,Mwl,M,Merr):
        #Gaussian error probablity for weak lensing mass 
        ans = gaussian(Mwl,M,Merr*M)
        return ans

