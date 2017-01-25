import numpy as np
from scipy import special
from sympy.functions import coth
import camb
from camb import model
import time
import cPickle as pickle

from Tinker_MF import dn_dlogM
from Tinker_MF import tinker_params

from orphics.tools.output import Plotter
from scipy.interpolate import interp1d

import szlibNumbafied as fast
from scipy.special import j0
from scipy.integrate import quad


def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print '%r %2.2f sec' % \
              (method.__name__,te-ts)
        return result

    return timed

def dictFromSection(config,sectionName):
    del config._sections[sectionName]['__name__']
    return dict([a, float(x)] for a, x in config._sections[sectionName].iteritems())

def listFromConfig(Config,section,name):
    return [float(x) for x in Config.get(section,name).split(',')]

class Cosmology(object):
    '''
    A wrapper around CAMB that tries to pre-calculate as much as possible
    Intended to be inherited by other classes like LimberCosmology and 
    ClusterCosmology
    '''
    def __init__(self,paramDict,constDict):

        cosmo = paramDict
        self.paramDict = paramDict
        c = constDict
        self.c = c

        self.c['TCMBmuK'] = self.c['TCMB'] * 1.0e6

        self.H0 = cosmo['H0']
        try:
            self.omch2 = cosmo['omch2']
        except:
            self.omch2 = (cosmo['om']-cosmo['ob'])*self.H0*self.H0/100./100.
            
        try:
            self.ombh2 = cosmo['ombh2']
        except:
            self.ombh2 = cosmo['ob']*self.H0*self.H0/100./100.
        
        self.pars = camb.CAMBparams()
        self.pars.set_cosmology(H0=self.H0, ombh2=self.ombh2, omch2=self.omch2)
        self.pars.InitPower.set_params(ns=cosmo['ns'],As=cosmo['As'])

        self.results= camb.get_background(self.pars)
        self.omnuh2 = self.pars.omegan * ((self.H0 / 100.0) ** 2.)
        

        self.rho_crit0 = 3. / (8. * np.pi) * (100 * 1.e5)**2. / c['G_CGS'] * c['MPC2CM'] / c['MSUN_CGS']


class ClusterCosmology(Cosmology):
    def __init__(self,paramDict,constDict,lmax):
        Cosmology.__init__(self,paramDict,constDict)
        self.om = paramDict['om']
        self.ol = paramDict['ol']
        self.rhoc0om = self.rho_crit0*self.om
        self.s8 = paramDict['s8']

        try:
            self.ells,self.cltt = pickle.load(open("output/cl"+time.strftime('%Y%m%d') +".pkl",'rb'))
        except:
            self.pars.set_accuracy(AccuracyBoost=2.0, lSampleBoost=4.0, lAccuracyBoost=4.0)
            self.pars.set_for_lmax(lmax=(lmax+500), lens_potential_accuracy=1, max_eta_k=2*(lmax+500))
            self.cltt =self.results.get_cmb_power_spectra(self.pars)['lensed_scalar'][2:,0]
            self.ells = np.arange(2,len(self.cltt)+2,1)
            self.cltt *= 2.*np.pi/self.ells/(self.ells+1.)
            self.ells = self.ells[self.ells<=lmax]
            self.cltt = self.cltt[self.ells<=lmax] 

            pickle.dump((self.ells,self.cltt) ,open("output/cl"+time.strftime('%Y%m%d') +".pkl",'wb'))
        self.clttfunc = interp1d(self.ells,self.cltt,fill_value=0.,bounds_error=False)

        
    def E_z(self,z):
        #hubble function
        ans = self.results.hubble_parameter(z)/self.paramDict['H0'] # 0.1% different from sqrt(om*(1+z)^3+ol)
        return ans

    def rhoc(self,z):
        #critical density as a function of z
        ans = self.rho_crit0*self.E_z(z)**2.
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
    def __init__(self,clusterCosmology):
        self.cc = clusterCosmology

    def pk(self,z_arr):
        #Using CAMB to get p of k
        
        
        self.cc.pars.set_matter_power(redshifts=z_arr, kmax=11.0)
        self.cc.pars.Transfer.high_precision = True
        
        self.cc.pars.NonLinear = model.NonLinear_none
        self.cc.results = camb.get_results(self.cc.pars)
        kh2, z2, pk2 = self.cc.results.get_matter_power_spectrum(minkh=2e-5, maxkh=11, npoints = 200,)
        s8 = np.array(self.cc.results.get_sigma8())
        return kh2, z2, pk2, s8
    
    def Halo_Tinker_test(self):
        
        #define parameters delta, M and z
        z_arr = np.array([0,0.8,1.6])
        M = 10**np.arange(10., 16, .1)
        delts = z_arr*0 + 200.
        delts_8 = z_arr*0 + 800.
        delts_32 = z_arr*0 + 3200.
    
        # start timer
        start = time.clock()
        
        #get p of k and s8 
        kh, z, pk, s8 = self.pk(z_arr)
        # dn_dlogM from tinker
        N = dn_dlogM(M,z_arr,self.cc.rhoc0om,delts,kh,pk[:1,:])
        N_8 = dn_dlogM(M,z_arr,self.cc.rhoc0om,delts_8,kh,pk[:1,:])
        N_32 = dn_dlogM(M,z_arr,self.cc.rhoc0om,delts_32,kh,pk[:1,:])
        
        elapsed1 = (time.clock() - start)
        #print elapsed1 
        
        #plot tinker values
        pl = Plotter()
        pl._ax.set_ylim(-3.6,-0.8)
        pl.add(np.log10(M),np.log10(N[:,0]*M/(self.cc.rhoc0om)),color='black')
        pl.add(np.log10(M),np.log10(N_8[:,0]*M/(self.cc.rhoc0om)),color='black')
        pl.add(np.log10(M),np.log10(N_32[:,0]*M/(self.cc.rhoc0om)),color='black')
        pl.done("output/tinkervals.png")
        
        return f
    
    def dVdz(self,z_arr):
        DA_z = self.cc.results.angular_diameter_distance(z_arr)
        dV_dz = DA_z**2 * (1.+z_arr)**2
        for i in xrange (len(z_arr)):
            dV_dz[i] /= (self.cc.results.h_of_z(z_arr[i]))
        #print (results.h_of_z(z_arr[i])),z_arr[i],100. * 1e5/2.99792458e10*hh
        dV_dz *= (self.cc.H0/100.)**3. # was h0
        return dV_dz

    def dn_dM(self,M,z_arr,delta):
        
        delts = z_arr*0 + delta
        kh, z, pk, s8 = self.pk(z_arr)
        fac = (self.cc.s8/s8[-1])**2 # sigma8 values are in reverse order
        pk *= fac
    
        dn_dlnm = dn_dlogM(M,z_arr,self.cc.rhoc0om,delts,kh,pk[:,:],'comoving')
        dn_dm = dn_dlnm/M
        return dn_dm
    
    def N_of_Mz(self,M,z_arr,delta):
        
        delts = z_arr*0 + delta
        kh, z, pk, s8 = self.pk(z_arr)
        fac = (self.cc.s8/s8[-1])**2 # sigma8 values are in reverse order
        pk *= fac
    
        dn_dlnm = dn_dlogM(M,z_arr,self.cc.rhoc0om,delts,kh,pk[:,:],'comoving')
        dn_dm = dn_dlnm/M
        dV_dz = self.dVdz(z_arr)
        
        N_dzdm = dn_dm[:,1:] * dV_dz[1:] * 4*np.pi
        return N_dzdm

    def N_of_z_SZ(self,z_arr,beams,noises,freqs,clusterDict,fileFunc=None,quick=True,tmaxN=5,numts=1000):


        h0 = 70.
        lnYmin = np.log(1e-13)
        dlnY = 0.1
        lnY = np.arange(lnYmin,lnYmin+10.,dlnY)
    
        Mexp = np.arange(14.0, 15.5, .1)
        #Mexp = np.arange(14.0, 15.4, 0.2)
        M = 10.**Mexp
        dM = np.gradient(M)
        rho_crit0m = self.cc.rhoc0om
        hh = h0/100.

        M200 = np.outer(M,np.zeros([len(z_arr)]))
        dM200 = np.outer(M,np.zeros([len(z_arr)]))
        P_func = np.outer(M,np.zeros([len(z_arr)]))
        sigN = np.outer(M,np.zeros([len(z_arr)]))
        M_arr =  np.outer(M,np.ones([len(z_arr)]))


        DA_z = self.cc.results.angular_diameter_distance(z_arr) * (self.cc.H0/100.)

        SZProf = SZ_Cluster_Model(self.cc,clusterDict,rms_noises = noises,fwhms=beams,freqs=freqs)



        for ii in xrange (len(z_arr)-1):
            i = ii + 1
            M200[:,i] = self.cc.Mass_con_del_2_del_mean200(M/hh,500,z_arr[i])
            dM200[:,i] = np.gradient(M200[:,i])
            for j in xrange(len(M)):
                try:
                    assert fileFunc is not None
                    filename = fileFunc(Mexp[j],z_arr[i])
                    print filename
                    sigN[j,i] = np.loadtxt(filename,unpack=True)[-1]
                except:
                    print "Calculating S/N because file not found or specified for M=",Mexp[j]," z=",z_arr[i]
                    if quick:
                        var = SZProf.quickVar(M[j],z_arr[i],tmaxN,numts)
                    else:
                        var = SZProf.filter_variance(M[j],z_arr[i])
                    sigN[j,i] = np.sqrt(var)
                print Mexp[j],z_arr[i]
            
            P_func[:,i] = SZProf.P_of_q (lnY,M_arr[:,i],z_arr[i],sigN[:,i])*dlnY

        #print P_func
        #dN_dzdm = self.N_of_Mz(M200,z_arr,200.)
        dn_dzdm = self.dn_dM(M200,z_arr,200.)
        #print dn_dzdm
        N_z = np.zeros(len(z_arr) - 1)
        for i in xrange (len(z_arr) - 1):
            N_z[i] = np.sum(dn_dzdm[:,i+1]*P_func[:,i+1]*dM200[:,i+1])

        #N_z = np.dot(dN_dzdm,np.transpose(P_func[:,1:]*dM200[:,1:]))

        return N_z

class SZ_Cluster_Model:
    def __init__(self,clusterCosmology,clusterDict,fwhms=[1.5],rms_noises =[1.], freqs = [150.],lmax=8000,dell=10,pmaxN=5,numps=1000,**options):
        self.cc = clusterCosmology
        self.P0 = clusterDict['P0']
        self.xc = clusterDict['xc']
        self.al = clusterDict['al']
        self.gm = clusterDict['gm']
        self.bt = clusterDict['bt']


        # build noise object
        self.dell = 10
        self.nlinv = 0.
        self.evalells = np.arange(2,lmax,self.dell)
        for freq,fwhm,noise in zip(freqs,fwhms,rms_noises):
            freq_fac = (self.f_nu(freq))**2


            nells = self.cc.clttfunc(self.evalells)+( self.noise_func(self.evalells,fwhm,noise) / self.cc.c['TCMBmuK']**2.)
            self.nlinv += (freq_fac)/nells

        self.nl = 1./self.nlinv


           
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
        self.gxrange = np.linspace(0.,pmaxN,numps)        
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



        if options.get("Model") == 'BBPS':
            self.P0 = 7.49
            self.xc = 0.710
            self.al = 1.0
            self.gm = -0.3
            self.bt = 4.19


    @timeit
    def quickVar(self,M,z,tmaxN=5.,numts=1000):




        # R500 in MPc, DAz in MPc, th500 in radians
        R500 = self.cc.rdel_c(M,z,500.).flatten()[0] # R500 in Mpc/h
        DAz = self.cc.results.angular_diameter_distance(z) * (self.cc.H0/100.) 
        th500 = R500/DAz

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


    
    def f_nu(self,nu):
        mu = self.cc.c['H_CGS']*(1e9*nu)/(self.cc.c['K_CGS']*self.cc.c['TCMB'])
        ans = mu*coth(mu/2.0) - 4.0
        return np.float(ans)
    
    def GNFW(self,xx,**options):
        ans = self.P0 / ((xx*self.xc)**self.gm * (1 + (xx*self.xc)**self.al)**((self.bt-self.gm)/self.al))
        if options.get("Model") == 'BBPS':
            ans = self.P0 * (xx/self.xc)**self.gm / (1 + (xx/self.xc)**self.al)**self.bt
        return ans
    
    def Prof(self,r,M,z,R500):
        R500 = R500
        xx = r / R500
        M_fac = M / (3e14) * (100./70.)
        P500 = 1.65e-3 * (100./70.)**2 * M_fac**(2./3.) * self.cc.E_z(z) #keV cm^3
        ans = P500 * self.GNFW(xx)
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

    def noise_func(self,ell,fwhm,rms_noise):        
        rms = rms_noise * (1./60.)*(np.pi/180.)
        tht_fwhm = np.deg2rad(fwhm / 60.)
        ans = (rms**2.) * np.exp((tht_fwhm**2.)*(ell**2.) / (8.*np.log(2.))) ## Add Hasselfield noise knee
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

        prof_int = 2.*np.pi*(np.sum((y2D_use*thta)[thta < 5*thtc])*self.dtht)**2
        
        return prof_int/var
    
    
    def Y_M(self,MM,zz):
        DA_z = self.cc.results.angular_diameter_distance(zz) * (self.cc.H0/100.)
        
        Y_star = 2.42e-10 #sterads
        #dropped h70 factor
        alpha_ym = 1.79
        b_ym = 0.8
        beta_ym = 0.66
        
        ans = Y_star*((b_ym)*MM/ 1e14)**alpha_ym *(DA_z/100.)**(-2.) * self.cc.E_z(zz) ** beta_ym
        #print (0.01/DA_z)**2
        return ans
    
    def Y_erf(self,Y_true,sigma_N):
        q = 5.
        sigma_Na = np.outer(sigma_N,np.ones(len(Y_true[0,:])))
        
        ans = 0.5 * (1. + special.erf((Y_true - q*sigma_Na)/(np.sqrt(2.)*sigma_Na)))
        return ans
    
    def P_of_Y (self,lnY,MM,zz):
        sig = 0.127  
        Y = np.exp(lnY)
        Ma = np.outer(MM,np.ones(len(Y[0,:])))
        numer = -1.*np.log(Y/self.Y_M(Ma,zz))**2
        ans = 1/(sig * np.sqrt(2*np.pi)) * np.exp(numer/(2.*sig))
        return ans
    
    def P_of_q (self,lnY,MM,zz,sigma_N):
        lnYa = np.outer(np.ones(len(MM)),lnY)
        
        DA_z = self.cc.results.angular_diameter_distance(zz) * (self.cc.H0/100.)
        
        sig_thresh = self.Y_erf(np.exp(lnYa),sigma_N)
        P_Y = self.P_of_Y(lnYa,MM, zz)

        ans = MM*0.0
        for ii in xrange(len(MM)):
            ans[ii] = np.sum(P_Y[ii,:]*sig_thresh[ii,:])
        return ans
