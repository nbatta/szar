from __future__ import print_function
from __future__ import division
from builtins import zip
from builtins import range
from builtins import object
from past.utils import old_div
import numpy as np
from scipy import special
from orphics.cosmology import noise_func
from szar.foregrounds import fgNoises, f_nu
from szar.counts import ClusterCosmology
from scipy.special import j0
from orphics.stats import timeit
from orphics import io
from scipy.interpolate import interp1d
import os

default_profile_params = {
    'P0': 8.403,
    'xc': 1.156,
    'al': 1.062,
    'bt': 5.4807,
    'gm': 0.3292}

def gnfw(xx, P0 = 8.403, xc = 1.156, gm = 0.3292, al = 1.062, bt = 5.4807):
    ans = old_div(P0, ((xx*xc)**gm * (1 + (xx*xc)**al)**(old_div((bt-gm),al))))
    return ans

def bbps(cc,M,z):
    R500 = old_div(cc.rdel_c(M,z,500.).flatten()[0], (old_div(cc.H0,100.))) # Mpc No hs
    DA_z = cc.results.angular_diameter_distance(z) # No hs
    M_fac = M / (3e14) * (old_div(100.,cc.H0))
    P500 = 1.65e-3 * (old_div(100.,cc.H0))**2 * M_fac**(old_div(2.,3.)) * self.cc.E_z(z) #keV cm^3
    intgrl = P500*np.sum(self.GNFW(old_div(rr,R500))*rr**2*np.sin(ell*rr/DA_z) / (ell*rr/DA_z) ) * dr
        
def gaussian(xx, mu, sig,noNorm=False):
    if (noNorm==True):
        return np.exp(-1.*(xx - mu)**2 / (2. * sig**2.))
    else:
        return 1./(sig * np.sqrt(2*np.pi)) * np.exp(-1.*(xx - mu)**2 / (2. * sig**2.))

def gaussian2Dnorm(sig_x,sig_y,rho):
    return sig_x*sig_y*2.0*np.pi*np.sqrt(1. - rho**2)

def gaussianMat2D(diff,sig_x,sig_y,rho):
    icov = old_div(np.matrix([[sig_y**2, -1.*sig_x*sig_y*rho],[-1.*sig_x*sig_y*rho, sig_x**2]]), (sig_x**2* sig_y**2*(1 - rho**2) )) 
    iC_diff = np.dot(icov,diff)
    expo = np.einsum('j...,j...->...',diff,iC_diff)
    ans = np.exp(-0.5 * expo)
    ans /= gaussian2Dnorm(sig_x,sig_y,rho)
    return ans

def gaussian2D(xx, mu_x, sig_x,yy,mu_y, sig_y, rho):
    exp0 = old_div(-1.,(2.*(1.-rho**2)))
    exp1 = old_div((xx - mu_x)**2, (sig_x**2.))
    exp2 = old_div((yy - mu_y)**2, (sig_y**2.))
    exp3 = 2*rho *(xx - mu_x)/sig_x *(yy - mu_y)/sig_y
    return 1./(sig_x*sig_y*2.0*np.pi*np.sqrt(1. - rho**2)) * np.exp(exp0*(exp1+exp2-exp3))

root_dir = os.path.dirname(os.path.realpath(__file__))+"/../"

class SZ_Cluster_Model(object):
    def __init__(self,clusterCosmology,clusterDict, \
                 fwhms=[1.5],rms_noises =[1.], freqs = [150.],lmax=8000,lknee=0.,alpha=1., \
                 dell=10,pmaxN=5,numps=1000,nMax=1, \
                 ymin=1.e-14,ymax=4.42e-9,dlnY = 0.1, qmin=5., \
                 ksz_file=root_dir+'input/ksz_BBPS.txt',ksz_p_file=root_dir+'input/ksz_p_BBPS.txt', \
                 tsz_cib_file=root_dir+'input/sz_x_cib_template.txt',fg=True,tsz_cib=False,
                 tsz_battaglia_template_csv=root_dir+"input/sz_template_battaglia.csv",v3mode=-1,fsky=None):

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
        fgs = fgNoises(self.cc.c,ksz_file=ksz_file,ksz_p_file=ksz_p_file,tsz_cib_file=tsz_cib_file,tsz_battaglia_template_csv=tsz_battaglia_template_csv)

        self.dell = dell
        self.nlinv = 0.
        self.nlinv_cmb = 0.
        self.nlinv_nofg = 0.
        self.nlinv_cmb_nofg = 0.
        self.evalells = np.arange(2,lmax,self.dell)


        if v3mode>-1:
            print("V3 flag enabled.")
            import szar.V3_calc_public as v3

            if v3mode <= 2:
                vfreqs = v3.Simons_Observatory_V3_LA_bands()
                print("Simons Obs")
                print("Replacing ",freqs,  " with ", vfreqs)
                freqs = vfreqs
                vbeams = v3.Simons_Observatory_V3_LA_beams()
                print("Replacing ",fwhms,  " with ", vbeams)
                fwhms = vbeams
                
                v3lmax = self.evalells.max()
                v3dell = np.diff(self.evalells)[0]
                print("Using ",fsky," for fsky")
                
                v3ell, N_ell_T_LA, N_ell_P_LA, Map_white_noise_levels = v3.Simons_Observatory_V3_LA_noise(sensitivity_mode=v3mode,f_sky=fsky,ell_max=v3lmax+v3dell,delta_ell=v3dell)
            elif v3mode == 3:
                vfreqs = v3.AdvACT_bands()
                print("AdvACT")
                print("Replacing ",freqs,  " with ", vfreqs)
                freqs = vfreqs
                vbeams = v3.AdvACT_beams()
                print("Replacing ",fwhms,  " with ", vbeams)
                fwhms = vbeams

                v3lmax = self.evalells.max()
                v3dell = np.diff(self.evalells)[0]
                print("Using ",fsky," for fsky")

                v3ell, N_ell_T_LA, N_ell_P_LA, Map_white_noise_levels = v3.AdvACT_noise(f_sky=fsky,ell_max=v3lmax+v3dell,delta_ell=v3dell)
            elif v3mode == 4:
                import szar.s4lat as s4
                mode = 2
                ncalc = s4.S4LatV1(mode, N_tels=2)
                vfreqs = ncalc.get_bands()
                print("S4")
                print("Replacing ",freqs,  " with ", vfreqs)
                freqs = vfreqs
                vbeams = ncalc.get_beams()
                print("Replacing ",fwhms,  " with ", vbeams)
                fwhms = vbeams

                v3lmax = self.evalells.max()
                v3dell = np.diff(self.evalells)[0]
                print("Using ",fsky," for fsky")

                v3ell, N_ell_T_LA_full, _ = ncalc.get_noise_curves(
                    fsky, v3lmax+v3dell, v3dell, full_covar=True, deconv_beam=True)
                N_ell_T_LA = np.diagonal(N_ell_T_LA_full).T
                print(N_ell_T_LA.shape)

            assert np.all(v3ell==self.evalells)
        
        # pl = io.Plotter(yscale='log',xlabel='l',ylabel='D_l')
            
        for ii,(freq,fwhm,noise) in enumerate(zip(freqs,fwhms,rms_noises)):
            freq_fac = (f_nu(self.cc.c,freq))**2


            if v3mode>-1:
                inst_noise = old_div(N_ell_T_LA[ii], self.cc.c['TCMBmuK']**2.)
            else:
                inst_noise = ( old_div(noise_func(self.evalells,fwhm,noise,lknee,alpha,dimensionless=False), self.cc.c['TCMBmuK']**2.))

            # pl.add(self.evalells,inst_noise*self.evalells**2.,color="C"+str(ii))
            # pl.add(self.evalells,N_ell_T_LA[ii]*self.evalells**2./ self.cc.c['TCMBmuK']**2.,color="C"+str(ii),ls="--")
            
            nells = self.cc.clttfunc(self.evalells)+inst_noise
            self.nlinv_nofg += old_div((freq_fac),nells)
            self.nlinv_cmb_nofg += (old_div(1.,inst_noise))

            totfg = (fgs.rad_ps(self.evalells,freq,freq) + fgs.cib_p(self.evalells,freq,freq) + \
                      fgs.cib_c(self.evalells,freq,freq) + fgs.ksz_temp(self.evalells)) \
                      / self.cc.c['TCMBmuK']**2. / ((self.evalells+1.)*self.evalells) * 2.* np.pi
            nells += totfg

            if (tsz_cib):
                tszcib = fgs.tSZ_CIB(self.evalells,freq,freq) \
                         / self.cc.c['TCMBmuK']**2. / ((self.evalells+1.)*self.evalells) * 2.* np.pi
                nells += tszcib

            self.nlinv += old_div((freq_fac),nells)
            self.nlinv_cmb += (old_div(1.,(inst_noise+totfg)))

        # pl.add(self.evalells,self.cc.clttfunc(self.evalells)*self.evalells**2.,color='k',lw=3)
        # pl.done(io.dout_dir+"v3comp.png")
        
        self.nl_old = old_div(1.,self.nlinv)
        self.nl_cmb = old_div(1.,self.nlinv_cmb)
        self.nl_nofg = old_div(1.,self.nlinv_nofg)
        self.nl_cmb_nofg = old_div(1.,self.nlinv_cmb_nofg)

        f_nu_tsz = f_nu(self.cc.c,np.array(freqs))
 
        if (len(freqs) > 1):
            fq_mat   = np.matlib.repmat(freqs,len(freqs),1)
            fq_mat_t = np.transpose(np.matlib.repmat(freqs,len(freqs),1))
        else:
            fq_mat   = np.array(freqs)
            fq_mat_t = np.array(freqs)

        self.nl = self.evalells*0.0

        for ii in range(len(self.evalells)):
            cmb_els = fq_mat*0.0 + self.cc.clttfunc(self.evalells[ii])

            if v3mode<0:
                inst_noise = ( old_div(noise_func(self.evalells[ii],np.array(fwhms),np.array(rms_noises),lknee,alpha,dimensionless=False), self.cc.c['TCMBmuK']**2.))
                nells = np.diag(inst_noise)
            elif v3mode<=2:
                ndiags = []
                for ff in range(len(freqs)):
                    inst_noise = old_div(N_ell_T_LA[ff,ii], self.cc.c['TCMBmuK']**2.)
                    ndiags.append(inst_noise)
                nells = np.diag(np.array(ndiags))
                # Adding in atmo. freq-freq correlations
                nells[0,1] = old_div(N_ell_T_LA[6,ii], self.cc.c['TCMBmuK']**2.)
                nells[1,0] = old_div(N_ell_T_LA[6,ii], self.cc.c['TCMBmuK']**2.)
                nells[2,3] = old_div(N_ell_T_LA[7,ii], self.cc.c['TCMBmuK']**2.)
                nells[3,2] = old_div(N_ell_T_LA[7,ii], self.cc.c['TCMBmuK']**2.)
                nells[4,5] = old_div(N_ell_T_LA[8,ii], self.cc.c['TCMBmuK']**2.)
                nells[5,4] = old_div(N_ell_T_LA[8,ii], self.cc.c['TCMBmuK']**2.)
            elif v3mode==3:
                ndiags = []
                for ff in range(len(freqs)):
                    inst_noise = old_div(N_ell_T_LA[ff,ii], self.cc.c['TCMBmuK']**2.)
                    ndiags.append(inst_noise)
                nells = np.diag(np.array(ndiags))
                # Adding in atmo. freq-freq correlations
                nells[0,1] = old_div(N_ell_T_LA[5,ii], self.cc.c['TCMBmuK']**2.)
                nells[1,0] = old_div(N_ell_T_LA[5,ii], self.cc.c['TCMBmuK']**2.)
                nells[2,3] = old_div(N_ell_T_LA[6,ii], self.cc.c['TCMBmuK']**2.)
                nells[3,2] = old_div(N_ell_T_LA[6,ii], self.cc.c['TCMBmuK']**2.)
                nells[3,4] = old_div(N_ell_T_LA[7,ii], self.cc.c['TCMBmuK']**2.)
                nells[4,3] = old_div(N_ell_T_LA[7,ii], self.cc.c['TCMBmuK']**2.)
            elif v3mode==4:
                nells = old_div(N_ell_T_LA_full[:,:,ii], self.cc.c['TCMBmuK']**2.)
                # ndiags = []
                # # Adding in atmo. freq-freq correlations
                # for ff in range(len(freqs)):
                #     for gg in range(len(freqs)):
                #         inst_noise = old_div(N_ell_T_LA_full[ff,gg,ii], self.cc.c['TCMBmuK']**2.)
                        
            totfg = (fgs.rad_ps(self.evalells[ii],fq_mat,fq_mat_t) + fgs.cib_p(self.evalells[ii],fq_mat,fq_mat_t) 
                     + fgs.cib_c(self.evalells[ii],fq_mat,fq_mat_t)) \
                     / self.cc.c['TCMBmuK']**2. / ((self.evalells[ii]+1.)*self.evalells[ii]) * 2.* np.pi

            if (tsz_cib):
                totfg += fgs.tSZ_CIB(self.evalells[ii],fq_mat,fq_mat_t) / self.cc.c['TCMBmuK']**2. / ((self.evalells[ii]+1.)*self.evalells[ii]) * 2.* np.pi
                totfg += fgs.tSZ(self.evalells[ii],fq_mat,fq_mat_t) / self.cc.c['TCMBmuK']**2. / ((self.evalells[ii]+1.)*self.evalells[ii]) * 2.* np.pi / 2. # factor of two accounts for resolved halos

            ksz = fq_mat*0.0 + fgs.ksz_temp(self.evalells[ii]) / self.cc.c['TCMBmuK']**2. / ((self.evalells[ii]+1.)*self.evalells[ii]) * 2.* np.pi

            nells += totfg + cmb_els + ksz

            self.nl[ii] = old_div(1.,(np.dot(np.transpose(f_nu_tsz),np.dot(np.linalg.inv(nells),f_nu_tsz))))

        # from orphics.io import Plotter
        # pl = Plotter(yscale='log',xlabel='l',ylabel='D_l')
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
        p = lambda x: old_div(1.,(((c*x)**gamma)*((1.+((c*x)**alpha))**(old_div((beta-gamma),alpha)))))

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
        th500 = old_div(R500,DAz)

        gnorm = 2.*np.pi*(th500**2.)*self.gnorm_pre

        u = lambda th: old_div(self.g(old_div(th,th500)),gnorm)
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
        var = old_div(1.,varinv)

        return var

    def GNFW(self,xx):
        ans = old_div(self.P0, ((xx*self.xc)**self.gm * (1 + (xx*self.xc)**self.al)**(old_div((self.bt-self.gm),self.al))))
        return ans

    def GNFWvar(self,xx,P0,xc,gm,al,bt):
        ans = old_div(P0, ((xx*xc)**gm * (1 + (xx*xc)**al)**(old_div((bt-gm),al))))
        return ans

    def Prof_tilde(self,ell,M,z):
        dr = 0.01
        R500 = self.cc.rdel_c(M,z,500.).flatten()[0] / (self.cc.H0/100.) # Mpc No hs
        DA_z = self.cc.results.angular_diameter_distance(z) # No hs
        rr = np.arange(dr,R500*5.0,dr)
        M_fac = M / (3e14) * (100./self.cc.H0)
        P500 = 1.65e-3 * (100./self.cc.H0)**2 * M_fac**(old_div(2.,3.)) * self.cc.E_z(z) #keV cm^3
        intgrl = P500*np.sum(self.GNFW(old_div(rr,R500))*rr**2*np.sin(ell*rr/DA_z) / (ell*rr/DA_z) ) * dr
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
        M_fac = M / (3e14) * (old_div(100.,70.))
        tau500 = 1.
        intgrl = tau500*np.sum(self.GNFWtau(old_div(rr,R500),P0,xc,gm,al,bt)*rr**2*np.sin(ell*rr/DA_z) / (ell*rr/DA_z) ) * dr
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
            P_func[:,i,:] = self.P_of_qn(lnY,M_arr[:,i],z_arr[i],sigN[:,i],q_arr)

        return P_func

    def Pfunc_qarr_corr(self,sigN,M,z_arr,q_arr,Mexp):#,mass_err):

        M_wl = 10**Mexp

        lnY = self.lnY

        P_func = np.zeros((M.size,z_arr.size,q_arr.size,M_wl.size))
        M_arr =  np.outer(M,np.ones([z_arr.size]))

        # P_func(M,z,q)
        for i in range(z_arr.size):
            for jj in range(M_wl.size):
                P_func[:,i,:,jj] = self.P_of_qn_corr(lnY,M_arr[:,i],z_arr[i],sigN[:,i],q_arr,M_wl[jj])#,mass_err[:,i])
        return P_func

    def Y_M(self,MM,zz):
        DA_z = self.cc.results.angular_diameter_distance(zz) * (self.cc.H0/100.)

        Y_star = self.scaling['Y_star'] #= 2.42e-10 #sterads
        #dropped h70 factor
        alpha_ym = self.scaling['alpha_ym']
        b_ym = self.scaling['b_ym']
        beta_ym = self.scaling['beta_ym']
        gamma_ym = self.scaling['gamma_ym']
        beta_fac = np.exp(beta_ym*(np.log(old_div(MM,1e14)))**2)

        ans = Y_star*((b_ym)*MM/ 1e14)**alpha_ym *(old_div(DA_z,100.))**(-2.) * beta_fac * self.cc.E_z(zz) ** (old_div(2.,3.)) * (1. + zz)**gamma_ym
        return ans

    def Y_erf(self,Y_true,sigma_N):
        q = self.qmin
        sigma_Na = np.outer(sigma_N,np.ones(len(Y_true[0,:])))
         
        ans = 0.5 * (1. + special.erf(old_div((Y_true - q*sigma_Na),(np.sqrt(2.)*sigma_Na))))
        return ans
    
    def P_of_Y (self,lnY,MM,zz):
 
        Y = np.exp(lnY)
        Ma = np.outer(MM,np.ones(len(Y[0,:])))
        Ysig = self.scaling['Ysig'] * (1. + zz)**self.scaling['gammaYsig'] * (old_div(Ma,1e14))**self.scaling['betaYsig']
        numer = -1.*(np.log(old_div(Y,self.Y_M(Ma,zz))))**2
        ans = 1./(Ysig * np.sqrt(2*np.pi)) * np.exp(old_div(numer,(2.*Ysig**2)))
        return ans

    def P_of_Y_corr (self,lnY,MM,zz,Mwl):
 
        Y = np.exp(lnY)
        Ma = np.outer(MM,np.ones(len(Y[0,:])))
        Ysig = self.scaling['Ysig'] * (1. + zz)**self.scaling['gammaYsig'] * (old_div(MM,1e14))**self.scaling['betaYsig']
        #Ysig = self.scaling['Ysig'] * (1. + zz)**self.scaling['gammaYsig'] * (Ma/1e14)**self.scaling['betaYsig']
        Msig = self.scaling['Msig'] * (1. + zz)**self.scaling['gammaMsig'] * (old_div(MM,1e14))**self.scaling['betaMsig']
        rho = self.scaling['rho_corr'] * (1. + zz)**self.scaling['gammarho'] * (old_div(MM,1e14))**self.scaling['betarho']
        diff_Y = np.log(old_div(Y,self.Y_M(Ma,zz)))
        diff_M = np.outer(np.log(Mwl*self.scaling['b_wl']/MM),np.ones(len(lnY[0,:])))
        diff_arr = np.array([diff_Y,diff_M])
        #Merr_arr = MM*0.0+Msig
        ans = Ma * 0.0

        for ii in range(len(MM)):
            ans[ii,:] = gaussianMat2D(diff_arr[:,ii,:],Ysig[ii],Msig[ii],rho[ii])
        return ans

    def P_of_q(self,lnY,MM,zz,sigma_N):

        lnYa = np.outer(np.ones(len(MM)),lnY)
        sig_thresh = self.Y_erf(np.exp(lnYa),sigma_N)
        P_Y = self.P_of_Y(lnYa,MM, zz)
        ans = np.trapz(P_Y*sig_thresh,lnY,np.diff(lnY),axis=1)
        return ans

    def P_of_qn(self,lnY,MM,zz,sigma_N,qarr):

        lnYa = np.outer(np.ones(len(MM)),lnY)
        
        P_Y = self.P_of_Y(lnYa,MM, zz)
        ans = np.zeros([len(MM),len(qarr)])
        
        for ii in range(len(qarr)):
            sig_thresh = self.q_prob(qarr[ii],lnYa,sigma_N)
            ans[:,ii] = np.trapz(P_Y*sig_thresh,lnY,np.diff(lnY),axis=1)
        return ans

    def P_of_qn_corr(self,lnY,MM,zz,sigma_N,qarr,Mwl):#,Merr):

        lnYa = np.outer(np.ones(len(MM)),lnY)
        P_Y = self.P_of_Y_corr(lnYa,MM,zz,Mwl)
        ans = np.zeros([len(MM),len(qarr)])
        for ii in range(len(qarr)):
            #sig_thresh = self.q_prob_corr(qarr[ii],lnYa,sigma_N,Mwl,MM,Merr)
            sig_thresh = self.q_prob(qarr[ii],lnYa,sigma_N)
            ans[:,ii] = np.trapz(P_Y*sig_thresh,lnY,np.diff(lnY),axis=1)
        return ans

    def q_prob (self,q_arr,lnY,sigma_N):
        #Gaussian error probablity for SZ S/N 
        sigma_Na = np.outer(sigma_N,np.ones(len(lnY[0,:])))
        Y = np.exp(lnY)
        ans = gaussian(q_arr,old_div(Y,sigma_Na),1.)
        return ans

    def q_prob_corr (self,q_arr,lnY,sigma_N,Mwl,MM,Merr):
        #Gaussian error probablity for SZ S/N and WL mass including correlated scatter
        rho = self.scaling['rho_corr']
        Y = np.exp(lnY)
        sigma_Na = np.outer(sigma_N,np.ones(len(lnY[0,:])))        
        diff_Y = (q_arr - old_div(Y,sigma_Na))
        diff_M = np.outer(Mwl*self.scaling['b_wl'] - MM,np.ones(len(lnY[0,:])))
        diff_arr = np.array([diff_Y,diff_M])
        Merr_arr = MM*Merr
        ans = sigma_Na * 0.0

        for ii in range(len(MM)):
            ans[ii,:] = gaussianMat2D(diff_arr[:,ii,:],1.,Merr_arr[ii],rho)
        return ans
    
    def Mwl_prob (self,Mwl,M,Merr):
        #Gaussian error probablity for weak lensing mass 
        ans = gaussian(Mwl*self.scaling['b_wl'],M,Merr*M)*self.scaling['b_wl']
        return ans

class Profiles(object):
    def __init__(self,clusterCosmology,fwhm=1.4):
        self.cc = clusterCosmology
        self.XH = 0.76 ### FIX THIS INTO CONSTANTS ###
        self.NNR = 100
        self.disc_fac = np.sqrt(2)
        self.l0 = 30000.
        self.fwhm = fwhm #### ARCMINS

        ### extrapolation variables
        self.inter_max = 5.
        self.inter_fine_fac = 4.

    #### JUST FOR TESTING ####
    def rho_sim_test(self,theta, x):
        fb = 0.0490086879038/0.314992203163
        rhoc = 2.77525e2
        Msol_cgs = 1.989e33
        kpc_cgs = 3.086e21
        a1,a2,a3 = theta
        gamma = 0.2
        ans = 10**a1 / ((x/0.5)**gamma * (1 + (x/0.5)**a2)**((a3 - gamma)/a2))
        ans *=  rhoc * Msol_cgs / kpc_cgs / kpc_cgs / kpc_cgs * fb 
        return ans
    
    def Pth_sim_test(self,theta,x):
        P0,xc,bt = theta
        al = 1.0
        gm = 0.3
        ans = P0 / ((x*xc)**gm * (1 + (x*xc)**al)**((bt-gm)/al))
        return ans
    #### JUST FOR TESTING ####

    def project_prof_sim_interpol(self,tht,Mvir,z,rho_sim,Pth_sim,test=False):

        theta_sim_rho = np.array([3.6337402156859753, 1.0369351928324118, 3.3290812595973063])
        theta_sim_pth = np.array([18.1, 0.5, 4.35])

        disc_fac = self.disc_fac 
        NNR = self.NNR
        drint = 1e-3 * (self.cc.c['MPC2CM'])
    
        AngDist = self.cc.results.angular_diameter_distance(z) * self.cc.H0/100.

        rvir = self.cc.rdel_c(Mvir,z,200)#/cc.c['MPC2CM']
        
        sig = 0
        sig2 = 0
        sig_p = 0
        sig2_p = 0
        area_fac = 0
        
        r_ext = AngDist*np.arctan(np.radians(tht/60.))
        r_ext2 = AngDist*np.arctan(np.radians(tht*disc_fac/60.))
        
        rad = (np.arange(1e4) + 1.0)/1e3 #in MPC
        rad2 = (np.arange(1e4) + 1.0)/1e3 #in MPC
        
        radlim = r_ext
        radlim2 = r_ext2
        
        dtht = np.arctan(radlim/AngDist)/NNR # rads
        dtht2 = np.arctan(radlim2/AngDist)/NNR # rads
        
        thta = (np.arange(NNR) + 1.)*dtht
        thta2 = (np.arange(NNR) + 1.)*dtht2
        
        for kk in xrange(NNR):
            rint = np.sqrt((rad)**2 + thta[kk]**2*AngDist**2)
            rint2 = np.sqrt((rad2)**2 + thta2[kk]**2*AngDist**2)
            
            if (test):
                sig += 2.0*np.pi*dtht*thta[kk]*np.sum(2.*self.rho_sim_test(theta_sim_rho,rint/rvir)*drint)
                sig2 += 2.0*np.pi*dtht2*thta2[kk]*np.sum(2.*self.rho_sim_test(theta_sim_rho,rint2/rvir)*drint)
            
                sig_p += 2.0*np.pi*dtht*thta[kk]*np.sum(2.*self.Pth_sim_test(theta_sim_pth,rint/rvir)*drint)
                sig2_p += 2.0*np.pi*dtht2*thta2[kk]*np.sum(2.*self.Pth_sim_test(theta_sim_pth,rint2/rvir)*drint)
            else:
                sig += 2.0*np.pi*dtht*thta[kk]*np.sum(2.*rho_sim(rint/rvir)*drint)
                sig2 += 2.0*np.pi*dtht2*thta2[kk]*np.sum(2.*rho_sim(rint2/rvir)*drint)
            
                sig_p += 2.0*np.pi*dtht*thta[kk]*np.sum(2.*Pth_sim(rint/rvir)*drint)
                sig2_p += 2.0*np.pi*dtht2*thta2[kk]*np.sum(2.*Pth_sim(rint2/rvir)*drint)

            area_fac += 2.0*np.pi*dtht*thta[kk]
            
        sig_all =(2*sig - sig2) * 1e-3 * self.cc.c['SIGMA_T'] * self.cc.c['TCMBmuK'] / self.cc.c['MP'] / (np.pi * np.radians(tht/60.)**2) # ((2. + 2.*self.XH)/(3.+5.*self.XH)) 
        sig_all_p = (2*sig_p - sig2_p) * self.cc.c['SIGMA_T']/(self.cc.c['ME']*self.cc.c['C']**2) / area_fac * \
            self.cc.c['TCMBmuK'] # muK # * ((2. + 2.*self.XH)/(3.+5.*self.XH))# muK
        
        sig_all /=(self.cc.H0/100.)
        sig_all_p /= (self.cc.H0/100.)

        return sig_all,sig_all_p

    def project_prof_beam_sim_interpol(self,tht,Mvir,z,rho_sim,Pth_sim,test=False):

        theta_sim_rho = np.array([3.6337402156859753, 1.0369351928324118, 3.3290812595973063])
        theta_sim_pth = np.array([18.1, 0.5, 4.35])
        
        fwhm = self.fwhm

        drint = 1e-3 * (self.cc.c['MPC2CM'])
        AngDist = self.cc.results.angular_diameter_distance(z) * self.cc.H0/100.
        disc_fac = self.disc_fac
        l0 = self.l0 
        NNR = self.NNR 
        NNR2 = 4*NNR
        
        fwhm *= np.pi / (180.*60.) #convert from arcmins to rads
        sigmaBeam = fwhm / np.sqrt(8.*np.log(2.))
        
        rvir = self.cc.rdel_c(Mvir,z,200)
        
        sig = 0
        sig2 = 0
        sig_p = 0
        sig2_p = 0
        area_fac = 0
        
        r_ext = AngDist*np.arctan(np.radians(tht/60.))
        r_ext2 = AngDist*np.arctan(np.radians(tht*disc_fac/60.))
        
        rad = (np.arange(1e4) + 1.0)/1e3 #in MPC/h
        rad2 = (np.arange(1e4) + 1.0)/1e3 #in MPC/h
        
        radlim = r_ext
        radlim2 = r_ext2
        
        dtht = np.arctan(radlim/AngDist)/NNR # rads
        dtht2 = np.arctan(radlim2/AngDist)/NNR # rads
        
        thta = (np.arange(NNR) + 1.)*dtht
        thta2 = (np.arange(NNR) + 1.)*dtht2
        
        thta_smooth = (np.arange(NNR2) + 1.)*dtht
        thta2_smooth = (np.arange(NNR2) + 1.)*dtht2
        
        rho2D = thta_smooth * 0.0
        rho2D2 = thta_smooth * 0.0
        Pth2D = thta_smooth * 0.0
        Pth2D2 = thta_smooth * 0.0
        
        rho2D_beam = thta * 0.0
        rho2D2_beam = thta* 0.0
        Pth2D_beam = thta* 0.0
        Pth2D2_beam = thta* 0.0
                 
        for kk in xrange(NNR2):
            rint  = np.sqrt((rad)**2  + thta_smooth[kk]**2 *AngDist**2)
            rint2 = np.sqrt((rad2)**2 + thta2_smooth[kk]**2*AngDist**2)

            if (test):
                rho2D[kk]  = np.sum(2.*self.rho_sim_test(theta_sim_rho,rint /rvir)*drint)
                rho2D2[kk] = np.sum(2.*self.rho_sim_test(theta_sim_rho,rint2/rvir)*drint)
                
                Pth2D[kk]  = np.sum(2.*self.Pth_sim_test(theta_sim_pth,rint /rvir)*drint)
                Pth2D2[kk] = np.sum(2.*self.Pth_sim_test(theta_sim_pth,rint2/rvir)*drint)
            else:
                rho2D[kk]  = np.sum(2.*rho_sim(rint /rvir)*drint)
                rho2D2[kk] = np.sum(2.*rho_sim(rint2/rvir)*drint)
                
                Pth2D[kk]  = np.sum(2.*Pth_sim(rint /rvir)*drint)
                Pth2D2[kk] = np.sum(2.*Pth_sim(rint2/rvir)*drint)
    
        for kk in xrange(NNR):
            rho2D_beam[kk]  = np.sum(thta_smooth  * rho2D  * np.exp(-0.5*thta_smooth**2 /sigmaBeam**2)  
                                     * special.iv(0, thta_smooth *thta[kk] / sigmaBeam**2))*dtht
            rho2D2_beam[kk] = np.sum(thta2_smooth * rho2D2 * np.exp(-0.5*thta2_smooth**2/sigmaBeam**2)
                                     * special.iv(0, thta2_smooth*thta2[kk]/ sigmaBeam**2))*dtht2

            Pth2D_beam[kk]  = np.sum(thta_smooth  * Pth2D  * np.exp(-0.5*thta_smooth**2 /sigmaBeam**2)  
                                     * special.iv(0, thta_smooth *thta[kk] / sigmaBeam**2))*dtht
            Pth2D2_beam[kk] = np.sum(thta2_smooth * Pth2D2 * np.exp(-0.5*thta2_smooth**2/sigmaBeam**2) 
                                     * special.iv(0, thta2_smooth*thta2[kk]/ sigmaBeam**2))*dtht2

            area_fac += 2.0*np.pi*dtht*thta[kk]
        
            rho2D_beam[kk]  *= np.exp(-0.5*thta[kk]**2 /sigmaBeam**2) / sigmaBeam**2
            rho2D2_beam[kk] *= np.exp(-0.5*thta2[kk]**2/sigmaBeam**2) / sigmaBeam**2
            Pth2D_beam[kk]  *= np.exp(-0.5*thta[kk]**2 /sigmaBeam**2) / sigmaBeam**2
            Pth2D2_beam[kk] *= np.exp(-0.5*thta2[kk]**2/sigmaBeam**2) / sigmaBeam**2

        sig  = 2.0*np.pi*dtht *np.sum(thta *rho2D_beam) 
        sig2 = 2.0*np.pi*dtht2*np.sum(thta2*rho2D2_beam) 

        sig_all_beam = (2*sig - sig2) * 1e-3 * self.cc.c['SIGMA_T'] * self.cc.c['TCMBmuK'] / self.cc.c['MP'] / (np.pi * np.radians(tht/60.)**2) # * ((2. + 2.*self.XH)/(3.+5.*self.XH)) 
        
        sig_p  = 2.0*np.pi*dtht*np.sum(thta*Pth2D_beam)
        sig2_p = 2.0*np.pi*dtht2*np.sum(thta2*Pth2D2_beam)
        
        sig_all_p_beam = (2*sig_p - sig2_p) * self.cc.c['SIGMA_T']/(self.cc.c['ME']*self.cc.c['C']**2) / area_fac * \
            self.cc.c['TCMBmuK'] # muK #* ((2. + 2.*self.XH)/(3.+5.*self.XH))# muK
    
        sig_all_beam /= (self.cc.H0/100.)
        sig_all_p_beam /= (self.cc.H0/100.) 

        return sig_all_beam, sig_all_p_beam

    def make_a_profile_test(self,thta_arc,M,z):
        rho = np.zeros(len(thta_arc))
        pth = np.zeros(len(thta_arc))
        for ii in xrange(len(thta_arc)):
            temp = self.project_prof_sim_interpol(thta_arc[ii],M,z,0,0,test=True)
            rho[ii] = temp[0]
            pth[ii] = temp[1]
        return rho,pth

    def make_a_obs_profile_test(self,thta_arc,M,z):
        rho = np.zeros(len(thta_arc))
        pth = np.zeros(len(thta_arc))
        for ii in xrange(len(thta_arc)):
            temp = self.project_prof_beam_sim_interpol(thta_arc[ii],M,z,0,0,test=True)
            rho[ii] = temp[0]
            pth[ii] = temp[1]
        return rho,pth

    def make_a_obs_profile_sim(self,thta_arc,M,z,rho_int,pres_int):
        rho = np.zeros(len(thta_arc))
        pth = np.zeros(len(thta_arc))
        for ii in xrange(len(thta_arc)):
            temp = self.project_prof_beam_sim_interpol(thta_arc[ii],M,z,rho_int,pres_int)
            rho[ii] = temp[0]
            pth[ii] = temp[1]
        return rho,pth

    def interpol_sim_profile(self,x,prof):
        #Including extrapolation
        if (np.max(x) < self.inter_max):
            fine_fac = self.inter_fine_fac
            xtr_inds = np.ceil(self.inter_max - np.max(x))
            xtr_inds = np.floor(self.inter_max - np.max(x))*fine_fac
            str_xtr = np.ceil(np.max(x)) + 1./fine_fac
            xtr = np.arange(xtr_inds)/fine_fac + str_xtr
            extend = np.poly1d(np.polyfit(x, np.log(prof), 3))

            ans = interp1d(np.append(x,xtr),np.append(prof,np.exp(extend(xtr))),kind='slinear',bounds_error=False,fill_value=0)
        else: 
            ans = interp1d(x,prof,kind='slinear',bounds_error=False,fill_value=0)
        return ans
