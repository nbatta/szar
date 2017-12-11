import numpy as np
from scipy import special
from orphics.tools.cmb import noise_func
from szar.foregrounds import fgNoises, f_nu
from szar.counts import ClusterCosmology
from scipy.special import j0
from orphics.tools.stats import timeit

def gaussian(xx, mu, sig):
    return 1./(sig * np.sqrt(2*np.pi)) * np.exp(-1.*(xx - mu)**2 / (2. * sig**2.))

def gaussian2Dnorm(sig_x,sig_y,rho):
    return sig_x*sig_y*2.0*np.pi*np.sqrt(1. - rho**2)

def gaussianMat2D(diff,sig_x,sig_y,rho):
    cov = np.array([[sig_x**2, sig_x*sig_y*rho],[sig_x*sig_y*rho, sig_y**2]])
    icov = np.linalg.inv(cov)
    ans = np.dot(np.transpose(diff),np.dot(icov,diff))
    ans /= gaussian2Dnorm(sig_x,sig_y,rho)
    return ans

def gaussian2D(xx, mu_x, sig_x,yy,mu_y, sig_y, rho):

    exp0 = -1./(2.*(1.-rho**2))
    exp1 = (xx - mu_x)**2 / (sig_x**2.)
    exp2 = (yy - mu_y)**2 / (sig_y**2.)
    exp3 = 2*rho *(xx - mu_x)/sig_x *(yy - mu_y)/sig_y
    return 1./(sig_x*sig_y*2.0*np.pi*np.sqrt(1. - rho**2)) * np.exp(exp0*(exp1+exp2-exp3))

class SZ_Cluster_Model:
    def __init__(self,clusterCosmology,clusterDict, \
                 fwhms=[1.5],rms_noises =[1.], freqs = [150.],lmax=8000,lknee=0.,alpha=1., \
                 dell=10,pmaxN=5,numps=1000,nMax=1, \
                 ymin=1.e-14,ymax=4.42e-9,dlnY = 0.1, qmin=5., \
                 ksz_file='input/ksz_BBPS.txt',ksz_p_file='input/ksz_p_BBPS.txt', \
                 tsz_cib_file='input/sz_x_cib_template.dat',fg=True,tsz_cib=False):

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

        fgs = fgNoises(self.cc.c,ksz_file=ksz_file,ksz_p_file=ksz_p_file,tsz_cib_file=tsz_cib_file,tsz_battaglia_template_csv="input/sz_template_battaglia.csv")

        self.dell = 10
        self.nlinv = 0.
        self.nlinv_cmb = 0.
        self.nlinv_nofg = 0.
        self.nlinv_cmb_nofg = 0.
        self.evalells = np.arange(2,lmax,self.dell)
        for freq,fwhm,noise in zip(freqs,fwhms,rms_noises):
            freq_fac = (f_nu(self.cc.c,freq))**2


            inst_noise = ( noise_func(self.evalells,fwhm,noise,lknee,alpha,dimensionless=False) / self.cc.c['TCMBmuK']**2.)
            nells = self.cc.clttfunc(self.evalells)+inst_noise
            self.nlinv_nofg += (freq_fac)/nells
            self.nlinv_cmb_nofg += (1./inst_noise)

            totfg = (fgs.rad_ps(self.evalells,freq,freq) + fgs.cib_p(self.evalells,freq,freq) + \
                      fgs.cib_c(self.evalells,freq,freq) + fgs.ksz_temp(self.evalells)) \
                      / self.cc.c['TCMBmuK']**2. / ((self.evalells+1.)*self.evalells) * 2.* np.pi
            nells += totfg

            if (tsz_cib):
                tszcib = fgs.tSZ_CIB(self.evalells,freq,freq) \
                         / self.cc.c['TCMBmuK']**2. / ((self.evalells+1.)*self.evalells) * 2.* np.pi
                nells += tszcib

            self.nlinv += (freq_fac)/nells
            self.nlinv_cmb += (1./(inst_noise+totfg))
        self.nl_old = 1./self.nlinv
        self.nl_cmb = 1./self.nlinv_cmb
        self.nl_nofg = 1./self.nlinv_nofg
        self.nl_cmb_nofg = 1./self.nlinv_cmb_nofg

        f_nu_tsz = f_nu(self.cc.c,np.array(freqs))
 
        if (len(freqs) > 1):
            fq_mat   = np.matlib.repmat(freqs,len(freqs),1)
            fq_mat_t = np.transpose(np.matlib.repmat(freqs,len(freqs),1))
        else:
            fq_mat   = freqs
            fq_mat_t = freqs

        self.nl = self.evalells*0.0

        for ii in range(len(self.evalells)):
            cmb_els = fq_mat*0.0 + self.cc.clttfunc(self.evalells[ii])
            inst_noise = ( noise_func(self.evalells[ii],np.array(fwhms),np.array(rms_noises),lknee,alpha,dimensionless=False) / self.cc.c['TCMBmuK']**2.)
            nells = np.diag(inst_noise)
            totfg = (fgs.rad_ps(self.evalells[ii],fq_mat,fq_mat_t) + fgs.cib_p(self.evalells[ii],fq_mat,fq_mat_t) 
                     + fgs.cib_c(self.evalells[ii],fq_mat,fq_mat_t)) \
                     / self.cc.c['TCMBmuK']**2. / ((self.evalells[ii]+1.)*self.evalells[ii]) * 2.* np.pi

            if (tsz_cib):
                totfg += fgs.tSZ_CIB(self.evalells[ii],fq_mat,fq_mat_t) / self.cc.c['TCMBmuK']**2. / ((self.evalells[ii]+1.)*self.evalells[ii]) * 2.* np.pi
                totfg += fgs.tSZ(self.evalells[ii],fq_mat,fq_mat_t) / self.cc.c['TCMBmuK']**2. / ((self.evalells[ii]+1.)*self.evalells[ii]) * 2.* np.pi / 2. # factor of two accounts for resolved halos

            ksz = fq_mat*0.0 + fgs.ksz_temp(self.evalells[ii]) / self.cc.c['TCMBmuK']**2. / ((self.evalells[ii]+1.)*self.evalells[ii]) * 2.* np.pi

            nells += totfg + cmb_els + ksz

            self.nl[ii] = 1./(np.dot(np.transpose(f_nu_tsz),np.dot(np.linalg.inv(nells),f_nu_tsz)))

        # from orphics.tools.io import Plotter
        # pl = Plotter(scaleY='log')
        # pl.add(self.evalells,self.nl*self.evalells**2.)
        # ells = np.arange(2,3000,1)
        # pl.add(ells,self.cc.clttfunc(ells)*ells**2.)
        # pl.done("nltt.png")
        # sys.exit()

        self.fg = fg

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

        if self.fg:
            noise = self.nl
        else:
            noise = self.nl_nofg
            
        varinv = np.trapz((integrands**2.)*ells*2.*np.pi/noise,ells,np.diff(ells))
        var = 1./varinv

        return var

    def GNFW(self,xx):
        ans = self.P0 / ((xx*self.xc)**self.gm * (1 + (xx*self.xc)**self.al)**((self.bt-self.gm)/self.al))
        return ans

    def GNFWvar(self,xx,P0,xc,gm,al,bt):
        ans = P0 / ((xx*xc)**gm * (1 + (xx*xc)**al)**((bt-gm)/al))
        return ans

    def Prof_tilde(self,ell,M,z):
        dr = 0.01
        R500 = self.cc.rdel_c(M,z,500.).flatten()[0] / (self.cc.H0/100.) # Mpc No hs
        DA_z = self.cc.results.angular_diameter_distance(z) # No hs
        rr = np.arange(dr,R500*5.0,dr)
        M_fac = M / (3e14) * (100./self.cc.H0)
        P500 = 1.65e-3 * (100./self.cc.H0)**2 * M_fac**(2./3.) * self.cc.E_z(z) #keV cm^3
        intgrl = P500*np.sum(self.GNFW(rr/R500)*rr**2*np.sin(ell*rr/DA_z) / (ell*rr/DA_z) ) * dr
        ans = 4.0*np.pi/DA_z**2 * intgrl
        ans *= self.cc.c['SIGMA_T']/(self.cc.c['ME']*self.cc.c['C']**2)*self.cc.c['MPC2CM']*self.cc.c['eV_2_erg']*1000.0
        #factor of 1000 to convert keV to eV

        return ans

    def Prof_tilde_tau(self,ell,M,z):

        P0 = 4e3
        xc = 0.5
        gm = 0.2
        al = 0.88
        bt = 3.83

        dr = 0.01
        R500 = self.cc.rdel_c(M,z,500.).flatten()[0]
        DA_z = self.cc.results.angular_diameter_distance(z)
        rr = np.arange(dr,R500*5.0,dr)
        M_fac = M / (3e14) * (100./70.)
        tau500 = 1.
        intgrl = tau500*np.sum(self.GNFWtau(rr/R500,P0,xc,gm,al,bt)*rr**2*np.sin(ell*rr/DA_z) / (ell*rr/DA_z) ) * dr
        ans = 4.0*np.pi/DA_z**2 * intgrl
        ans *= self.cc.c['SIGMA_T'] # CHECK Units
        #/(self.cc.c['ME']*self.cc.c['C']**2)*self.cc.c['MPC2CM']*self.cc.c['eV_2_erg']*1000.0

        return ans

    def Pfunc(self,sigN,M,z_arr):

        lnY = self.lnY

        P_func = np.outer(M,np.zeros([len(z_arr)]))
        M_arr =  np.outer(M,np.ones([len(z_arr)]))

        for i in range(z_arr.size):
            P_func[:,i] = self.P_of_q(lnY,M_arr[:,i],z_arr[i],sigN[:,i])
            

        return P_func

    def Pfunc_qarr(self,sigN,M,z_arr,q_arr):

        lnY = self.lnY

        P_func = np.zeros((M.size,z_arr.size,q_arr.size))
        M_arr =  np.outer(M,np.ones([z_arr.size]))

        # P_func(M,z,q)
        for i in range(z_arr.size):
            for kk in range(q_arr.size):
                P_func[:,i,kk] = self.P_of_qn(lnY,M_arr[:,i],z_arr[i],sigN[:,i],q_arr[kk])

        return P_func

    def Pfunc_qarr_corr(self,sigN,M,z_arr,q_arr,Mexp,mass_err):

        M_wl = 10**Mexp

        lnY = self.lnY

        P_func = np.zeros((M.size,z_arr.size,q_arr.size,M_wl.size))
        M_arr =  np.outer(M,np.ones([z_arr.size]))

        # P_func(M,z,q)
        for i in range(z_arr.size):
            for kk in range(q_arr.size):
                for jj in range(M_wl.size):
                    P_func[:,i,kk,jj] = self.P_of_qn_corr(lnY,M_arr[:,i],z_arr[i],sigN[:,i],q_arr[kk],M_wl[jj],mass_err[:,i])
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
        Ysig = self.scaling['Ysig'] * (1. + zz)**self.scaling['gammaYsig'] * (Ma/1e14)**self.scaling['betaYsig']
        numer = -1.*(np.log(Y/self.Y_M(Ma,zz)))**2
        ans = 1./(Ysig * np.sqrt(2*np.pi)) * np.exp(numer/(2.*Ysig**2))
        return ans

    def P_of_q(self,lnY,MM,zz,sigma_N):
        lnYa = np.outer(np.ones(len(MM)),lnY)

        sig_thresh = self.Y_erf(np.exp(lnYa),sigma_N)
        P_Y = self.P_of_Y(lnYa,MM, zz)

        ans = MM*0.0
        for ii in range(len(MM)):
            ans[ii] = np.trapz(P_Y[ii,:]*sig_thresh[ii,:],lnY,np.diff(lnY))
        return ans

    def P_of_qn(self,lnY,MM,zz,sigma_N,qarr):
        lnYa = np.outer(np.ones(len(MM)),lnY)

        sig_thresh = self.q_prob(qarr,lnYa,sigma_N)
        P_Y = self.P_of_Y(lnYa,MM, zz)
        ans = MM*0.0
        for ii in range(len(MM)):
            ans[ii] = np.trapz(P_Y[ii,:]*sig_thresh[ii,:],lnY,np.diff(lnY))
        return ans

    def P_of_qn_corr(self,lnY,MM,zz,sigma_N,qarr,Mwl,Merr):
        lnYa = np.outer(np.ones(len(MM)),lnY)
        sig_thresh = self.q_prob_corr(qarr,lnYa,sigma_N,Mwl,MM,Merr)
        P_Y = self.P_of_Y(lnYa,MM, zz)
        ans = sig_thresh*0.0
        for ii in range(len(MM)):
            ans[ii] = np.trapz(P_Y[ii,:]*sig_thresh[ii,:],lnY,np.diff(lnY))
        return ans

    def q_prob (self,q_arr,lnY,sigma_N):
        #Gaussian error probablity for SZ S/N 
        sigma_Na = np.outer(sigma_N,np.ones(len(lnY[0,:])))
        Y = np.exp(lnY)
        ans = gaussian(q_arr,Y/sigma_Na,1.)
        return ans

    def q_prob_corr (self,q_arr,lnY,sigma_N,Mwl,MM,Merr):
        #Gaussian error probablity for SZ S/N
        rho = self.scaling['rho_corr']
        sigma_Na = np.outer(sigma_N,np.ones(len(lnY[0,:])))
        Mwla = np.outer(Mwl)
        Y = np.exp(lnY)
        print("size")
        print((Y.size, Mwla.size, MM.size)) 
        
        diff_Y = q_arr - Y/sigma_Na
        diff_M = diff_Y*0.0 + Mwl*self.scaling['b_wl'] - MM
        diff_arr = np.array([diff_Y,diff_M])
        ans = gaussianMat2D(diff_arr,1.,Merr*MM,rho)
        #cov = np.array([[1.,rho*Merr*MM],[rho*Merr*MM (Merr*MM)**2 ]])
        #covi = np.linalg.inv(cov)

        #ans = np.dot(np.transpose(diff_arr),np.dot(covi,diff_arr))
        #norm = gaussian2D_norm(1,Merra*MMa,rho)
        #ans = gaussian2D(q_arr,Y/sigma_Na,1.,Mwl*self.scaling['b_wl'],MMa,Merra*MMa,rho)
        return ans
    
    def Mwl_prob (self,Mwl,M,Merr):
        #Gaussian error probablity for weak lensing mass 
        ans = gaussian(Mwl*self.scaling['b_wl'],M,Merr*M)*self.scaling['b_wl']
        return ans

    # def Mwl_prob (self,Mwl,M,Merr):
    #     #Gaussian error probablity for weak lensing mass 
    #     ans = gaussian(Mwl,M,(Merr+0.01)*M) #* gaussian(Mwl,M,0.01*M)
    #     return ans

