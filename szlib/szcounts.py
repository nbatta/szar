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
    Nmz = np.trapz(Nmzq,dx=np.diff(qbin_edges),axis=2)
    Nofz = np.trapz(Nmz.T,dx=np.diff(10**mexp_edges),axis=1)
    N = np.trapz(Nofz.T,dx=np.diff(z_edges))
    if returnNz:
        return N,Nofz
    else:
        return N


def getNmzq(Nmzq,mexp_edges,z_edges,q_edges):
    """Get number of clusters in each (m,z,q) bin 
    given dN/DmDqDz and the corresponding 
    log10(mass), z and q grid edges
    """
    # Ndq = np.multiply(Nmzq,np.diff(q_edges))
    # Ndz = np.multiply(Nmzq,np.diff(z_edges))
    # Ndm = np.multiply(Nmzq,np.diff(10**mexp_edges))
    Ndq = np.multiply(Nmzq,np.diff(q_edges).reshape((1,1,q_edges.size-1)))
    Ndz = np.multiply(Ndq,np.diff(z_edges).reshape((1,z_edges.size-1,1)))
    Ndm = np.multiply(Ndz,np.diff(10**mexp_edges).reshape((mexp_edges.size-1,1,1)))
    return Ndm
    

def gaussian(xx, mu, sig):
    return 1./(sig * np.sqrt(2*np.pi)) * np.exp(-1.*(xx - mu)**2 / (2. * sig**2.))

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
    
    # def rhoc_alt(self,z):
    #     #critical density as a function of z
    #     ans = self.rho_crit0*self.E_z(z)**2.
    #     return ans

    # def rdel_c_alt(self,Moverh,z,delta):
    #     #spherical overdensity radius w.r.t. the critical density
    #     rhocz = self.rhoc_alt(z)
    #     M = Moverh/self.h
    #     ans = (3. * M / (4. * np.pi * delta*rhocz))**(1.0/3.0)
    #     return ans*self.h
        
    
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
        self.DAz = self.cc.results.angular_diameter_distance(self.zarr)        
        self._initdVdz(self.zarr)

        self.sigN = None
        self.YM = None
        self.Pfunc_qarr = None
        self.Pfunc = None

        M_edges = 10**Mexp_edges
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


        # PK = camb.get_matter_power_interpolator(self.cc.pars, nonlinear=False, kmax=kmax, zs=zarr)
        # kh = np.linspace(kmin,kmax,knum)
        # powerZK = np.zeros((zarr.size,kh.size))
        # for i,z in enumerate(zarr):
        #     powerZK[i,:] = PK.P(z,kh)

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
            #N_z[i] = np.trapz(dn_dzdm[:,i],self.M200[:,i],np.diff(self.M200[:,i]))
            N_z[i] = np.dot(dn_dzdm[:,i],np.diff(self.M200_edges[:,i]))

        return N_z*4.*np.pi

    @timeit
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
         self.Pfunc = SZCluster.Pfunc(self.sigN.copy(),self.M,self.zarr)

    def updatePfunc_qarr(self,SZCluster,q_arr):
        print "Calculating P_func_qarr. This takes a while..."
        self.Pfunc_qarr = SZCluster.Pfunc_qarr(self.sigN.copy(),self.M,self.zarr,q_arr)

    def N_of_z_SZ(self,SZCluster):
        # this is dN/dz(z) with selection

        z_arr = self.zarr
        
        if self.sigN is None: self.updateSigN(SZCluster)
        if self.Pfunc is None: self.updatePfunc(SZCluster)
        P_func = self.Pfunc

        dn_dzdm = self.dn_dM(self.M200,200.)
        N_z = np.zeros(z_arr.size)
        for i in xrange (z_arr.size):
            #N_z[i] = np.trapz(dn_dzdm[:,i]*P_func[:,i],self.M200[:,i],np.diff(self.M200[:,i]))
            N_z[i] = np.dot(dn_dzdm[:,i]*P_func[:,i],np.diff(self.M200_edges[:,i]))


        return N_z* self.dVdz[:]*4.*np.pi


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
            #N_z[i] = np.trapz(dn_dVdm[:,i]*P_func[:,i] / (mass_err[:,i]**2 + alpha_ym**2 * (self.sigN[:,i]/self.YM[:,i])**2),self.M200[:,i])
            #N_tot_z[i] = np.trapz(dn_dVdm[:,i]*P_func[:,i],self.M200[:,i])
            N_z[i] = np.dot(dn_dVdm[:,i]*P_func[:,i] / (mass_err[:,i]**2 + alpha_ym**2 * (self.sigN[:,i]/self.YM[:,i])**2),np.diff(self.M200_edges[:,i]))
            N_tot_z[i] = np.dot(dn_dVdm[:,i]*P_func[:,i],np.diff(self.M200_edges[:,i]))
        #err_WL_mass = 4.*np.pi* fsky*np.trapz(N_z*dV_dz[:],self.zarr)
        #Ntot = 4.*np.pi* fsky*np.trapz(N_tot_z*dV_dz[:],self.zarr)
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

        # \int dm  dn/dzdm
        for kk in xrange(q_arr.size):
            for jj in xrange(m_wl.size):
                for i in xrange (z_arr.size):
                    dM = np.diff(self.M200_edges[:,i])
                    dNdzmq[jj,i,kk] = np.dot(dn_dVdm[:,i]*P_func[:,i,kk]*SZCluster.Mwl_prob(10**(m_wl[jj]),M_arr[:,i],mass_err[:,i]),dM) * dV_dz[i]*4.*np.pi
        
            
        return dNdzmq

class SZ_Cluster_Model:
    def __init__(self,clusterCosmology,clusterDict,fwhms=[1.5],rms_noises =[1.], freqs = [150.],lmax=8000,lknee=0.,alpha=1.,dell=10,pmaxN=5,numps=1000,nMax=1,ymin=1.e-14,ymax=4.42e-9,dlnY = 0.1,qmin=6.):
        self.cc = clusterCosmology
        self.P0 = clusterDict['P0']
        self.xc = clusterDict['xc']
        self.al = clusterDict['al']
        self.gm = clusterDict['gm']
        self.bt = clusterDict['bt']

        self.scaling = self.cc.paramDict

        self.qmin = qmin

        lnYmin = np.log(ymin)
        lnYmax = np.log(ymax)
        self.lnY = np.arange(lnYmin,lnYmax,dlnY)

        # lnYmin = np.log(1e-13)
        # dlnY = 0.1
        # lnY = np.arange(lnYmin,lnYmin+10.,dlnY)
        
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
        # pl.add(ls,self.nl*ls**2.,label="fg")
        # pl.add(ls,self.nl2*ls**2.,label="no fg")
        # # pl.add(ls,self.nl*freq_fac*ls**2.,label="freq")
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

        self.gnorm_pre = np.trapz(self.gxrange*self.gint,self.gxrange)

        # pl = Plotter()
        # pl.add(gxrange,gint)
        # pl.done("output/gint.png")


    def quickVar(self,M,z,tmaxN=5.,numts=1000):




        # R500 in MPc, DAz in MPc, th500 in radians
        R500 = self.cc.rdel_c(M,z,500.).flatten()[0] # R500 in Mpc/h
        DAz = self.cc.results.angular_diameter_distance(z) * (self.cc.H0/100.) 
        th500 = R500/DAz
        # gnorm = 2pi th500^2  \int dx x g(x)
        gnorm = 2.*np.pi*(th500**2.)*self.gnorm_pre 

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


    def Pfunc(self,sigN,M,z_arr):

        lnY = self.lnY

        P_func = np.outer(M,np.zeros([len(z_arr)]))
        M_arr =  np.outer(M,np.ones([len(z_arr)]))


        for i in xrange(z_arr.size):
            P_func[:,i] = self.P_of_q(lnY,M_arr[:,i],z_arr[i],sigN[:,i])

        return P_func

    @timeit
    def Pfunc_qarr(self,sigN,M,z_arr,q_arr):

        lnY = self.lnY

        P_func = np.zeros((M.size,z_arr.size,q_arr.size))
        M_arr =  np.outer(M,np.ones([z_arr.size]))


        # P_func(M,z,q)
        for i in xrange(z_arr.size):
            for kk in xrange(q_arr.size):
                P_func[:,i,kk] = self.P_of_qn(lnY,M_arr[:,i],z_arr[i],sigN[:,i],q_arr[kk])


        return P_func

    

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
        q = self.qmin
        sigma_Na = np.outer(sigma_N,np.ones(len(Y_true[0,:])))
        
        ans = 0.5 * (1. + special.erf((Y_true - q*sigma_Na)/(np.sqrt(2.)*sigma_Na)))
        return ans
    
    def P_of_Y (self,lnY,MM,zz):

        Y = np.exp(lnY)
        Ma = np.outer(MM,np.ones(len(Y[0,:])))
        Ysig = self.scaling['Ysig'] * (1. + zz)**self.scaling['gammaYsig'] * (Ma/(self.cc.H0/100.)/6e14)**self.scaling['betaYsig']
        numer = -1.*(np.log(Y/self.Y_M(Ma,zz)))**2
        ans = 1./(Ysig * np.sqrt(2*np.pi)) * np.exp(numer/(2.*Ysig**2))
        return ans
    
    def P_of_q(self,lnY,MM,zz,sigma_N):
        lnYa = np.outer(np.ones(len(MM)),lnY)
        
        sig_thresh = self.Y_erf(np.exp(lnYa),sigma_N)
        P_Y = self.P_of_Y(lnYa,MM, zz)

        ans = MM*0.0
        for ii in xrange(len(MM)):
            ans[ii] = np.trapz(P_Y[ii,:]*sig_thresh[ii,:],lnY,np.diff(lnY))
        return ans

    def P_of_qn(self,lnY,MM,zz,sigma_N,qarr):
        lnYa = np.outer(np.ones(len(MM)),lnY)
        
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
        #intgrl = ell *0.0
        #for ii in xrange(len(ell)):
        #    intgrl[ii] = np.sum(self.Prof(rr,M,z)*rr**2*np.sinc(ell[ii]*rr/DA_z)) * dr
        intgrl = np.sum(self.Prof(rr,M,z)*rr**2*np.sin(ell*rr/DA_z) / (ell*rr/DA_z) ) * dr
        ans = 4.0*np.pi/DA_z**2 * intgrl
        ans *= c.SIGMA_T/(c.ME*c.C**2)*c.MPC2CM*c.eV_2_erg*1000.0
        return ans
    
