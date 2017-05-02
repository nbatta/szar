import numpy as np
from scipy import special
from orphics.tools.cmb import noise_func
from szar.foregrounds import fgNoises
from szar.counts import ClusterCosmology,f_nu
from scipy.special import j0
from orphics.tools.stats import timeit

def gaussian(xx, mu, sig):
    return 1./(sig * np.sqrt(2*np.pi)) * np.exp(-1.*(xx - mu)**2 / (2. * sig**2.))

class SZ_Cluster_Model:
    def __init__(self,clusterCosmology,clusterDict, \
                 fwhms=[1.5],rms_noises =[1.], freqs = [150.],lmax=8000,lknee=0.,alpha=1., \
                 dell=10,pmaxN=5,numps=1000,nMax=1, \
                 ymin=1.e-14,ymax=4.42e-9,dlnY = 0.1, \
                 qmin=6., \
                 ksz_file='input/ksz_BBPS.txt',ksz_p_file='input/ksz_p_BBPS.txt'):

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

        fgs = fgNoises(self.cc.c,ksz_file=ksz_file,ksz_p_file=ksz_p_file)

        self.dell = 10
        self.nlinv = 0.
        self.nlinv2 = 0.
        self.evalells = np.arange(2,lmax,self.dell)
        for freq,fwhm,noise in zip(freqs,fwhms,rms_noises):
            freq_fac = (f_nu(self.cc.c,freq))**2


            nells = self.cc.clttfunc(self.evalells)+( noise_func(self.evalells,fwhm,noise,lknee,alpha) / self.cc.c['TCMBmuK']**2.)
            self.nlinv2 += (freq_fac)/nells

            nells += (fgs.rad_ps(self.evalells,freq,freq) + fgs.cib_p(self.evalells,freq,freq) + \
                      fgs.cib_c(self.evalells,freq,freq) + fgs.ksz_temp(self.evalells)) \
                      / self.cc.c['TCMBmuK']**2. / ((self.evalells+1.)*self.evalells) * 2.* np.pi

            self.nlinv += (freq_fac)/nells
        self.nl = 1./self.nlinv
        self.nl2 = 1./self.nlinv2

        c = self.xc
        alpha = self.al
        beta = self.bt
        gamma = self.gm
        p = lambda x: 1./(((c*x)**gamma)*((1.+((c*x)**alpha))**((beta-gamma)/alpha)))

        pmaxN = pmaxN
        numps = numps
        pzrange = np.linspace(-pmaxN,pmaxN,numps)
        self.g = lambda x: np.trapz(p(np.sqrt(pzrange**2.+x**2.)),pzrange,np.diff(pzrange))

        self.gxrange = np.linspace(0.,nMax,numps)
        self.gint = np.array([self.g(x) for x in self.gxrange])

        self.gnorm_pre = np.trapz(self.gxrange*self.gint,self.gxrange)
    #@timeit
    def quickVar(self,M,z,tmaxN=5.,numts=1000):

        R500 = self.cc.rdel_c(M,z,500.).flatten()[0] # R500 in Mpc/h 
        DAz = self.cc.results.angular_diameter_distance(z) * (self.cc.H0/100.)
        th500 = R500/DAz

        gnorm = 2.*np.pi*(th500**2.)*self.gnorm_pre

        u = lambda th: self.g(th/th500)/gnorm
        thetamax = tmaxN * th500
        thetas = np.linspace(0.,thetamax,numts)
        uint = np.array([u(t) for t in thetas])

        ells = self.evalells
        integrand = lambda l: np.trapz(j0(l*thetas)*uint*thetas,thetas,np.diff(thetas))
        integrands = np.array([integrand(ell) for ell in ells])

        varinv = np.trapz((integrands**2.)*ells*2.*np.pi/self.nl,ells,np.diff(ells))
        var = 1./varinv

        return var

    def GNFW(self,xx):
        ans = self.P0 / ((xx*self.xc)**self.gm * (1 + (xx*self.xc)**self.al)**((self.bt-self.gm)/self.al))
        return ans

    def Prof_tilde(self,ell,M,z):
        dr = 0.01
        R500 = self.cc.rdel_c(M,z,500.).flatten()[0]
        DA_z = self.cc.results.angular_diameter_distance(z)
        rr = np.arange(dr,R500*5.0,dr)
        M_fac = M / (3e14) * (100./70.)
        P500 = 1.65e-3 * (100./70.)**2 * M_fac**(2./3.) * self.cc.E_z(z) #keV cm^3
        intgrl = P500*np.sum(self.GNFW(rr/R500)*rr**2*np.sin(ell*rr/DA_z) / (ell*rr/DA_z) ) * dr
        ans = 4.0*np.pi/DA_z**2 * intgrl
        ans *= self.cc.c['SIGMA_T']/(self.cc.c['ME']*self.cc.c['C']**2)*self.cc.c['MPC2CM']*self.cc.c['eV_2_erg']*1000.0

        return ans

    def Pfunc(self,sigN,M,z_arr):

        lnY = self.lnY

        P_func = np.outer(M,np.zeros([len(z_arr)]))
        M_arr =  np.outer(M,np.ones([len(z_arr)]))

        for i in xrange(z_arr.size):
            P_func[:,i] = self.P_of_q(lnY,M_arr[:,i],z_arr[i],sigN[:,i])
            

        return P_func

    def Pfunc_qarr(self,sigN,M,z_arr,q_arr):

        lnY = self.lnY

        P_func = np.zeros((M.size,z_arr.size,q_arr.size))
        M_arr =  np.outer(M,np.ones([z_arr.size]))

        # P_func(M,z,q)
        for i in xrange(z_arr.size):
            for kk in xrange(q_arr.size):
                P_func[:,i,kk] = self.P_of_qn(lnY,M_arr[:,i],z_arr[i],sigN[:,i],q_arr[kk])

        return P_func

    def Y_M(self,MM,zz):
        DA_z = self.cc.results.angular_diameter_distance(zz) * (self.cc.H0/100.)

        Y_star = self.scaling['Y_star'] #= 2.42e-10 #sterads
        #dropped h70 factor
        alpha_ym = self.scaling['alpha_ym']
        b_ym = self.scaling['b_ym']
        beta_ym = self.scaling['beta_ym']
        gamma_ym = self.scaling['gamma_ym']
        beta_fac = np.exp(beta_ym*(np.log(MM/1e14))**2)

        ans = Y_star*((b_ym)*MM/ 1e14)**alpha_ym *(DA_z/100.)**(-2.) * beta_fac * self.cc.E_z(zz) ** (2./3.) * (1. + zz)**gamma_ym
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
        ans = gaussian(Mwl*self.scaling['b_wl'],M,Merr*M)
        return ans
