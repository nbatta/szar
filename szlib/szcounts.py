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
from orphics.theory.cosmology import Cosmology
from orphics.tools.stats import timeit
from scipy.interpolate import interp1d
from orphics.analysis.flatMaps import interpolateGrid

import szlib.szlibNumbafied as fast
from scipy.special import j0

def f_nu(cc,nu):
    mu = cc.c['H_CGS']*(1e9*nu)/(cc.c['K_CGS']*cc.c['TCMB'])
    ans = mu*coth(mu/2.0) - 4.0
    return np.float(ans)


class ClusterCosmology(Cosmology):
    def __init__(self,paramDict,constDict,lmax):
        Cosmology.__init__(self,paramDict,constDict,lmax)
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
        #dV/dzdOmega
        DA_z = self.cc.results.angular_diameter_distance(z_arr)
        dV_dz = DA_z**2 * (1.+z_arr)**2
        for i in xrange (len(z_arr)):
            dV_dz[i] /= (self.cc.results.h_of_z(z_arr[i]))
        #print (results.h_of_z(z_arr[i])),z_arr[i],100. * 1e5/2.99792458e10*hh
        dV_dz *= (self.cc.H0/100.)**3. # was h0
        return dV_dz

    def dn_dM(self,M,z_arr,delta):
        #Mass Function
        delts = z_arr*0 + delta
        kh, z, pk, s8 = self.pk(z_arr)
        #fac = (self.cc.s8/s8[-1])**2 # sigma8 values are in reverse order
        #pk *= fac
        print "s8", np.max(s8)
    
        dn_dlnm = dn_dlogM(M,z_arr,self.cc.rhoc0om,delts,kh,pk[:,:],'comoving')
        dn_dm = dn_dlnm/M
        return dn_dm
    
    def N_of_Mz(self,M,z_arr,delta):
        
        dn_dm = self.dn_dM(M,z_arr,delta)
        dV_dz = self.dVdz(z_arr)
        N_dzdm = dn_dm[:,1:] * dV_dz[1:]
        return N_dzdm

    def N_of_z(self,z_arr):

        Mexp = np.arange(13.5, 15.71, .1)
        M = 10.**Mexp
        M200 = np.outer(M,np.zeros([len(z_arr)]))

        for ii in xrange (len(z_arr)-1):
            i = ii + 1
            M200[:,i] = self.cc.Mass_con_del_2_del_mean200(M,500,z_arr[i])

        dn_dzdm = self.N_of_Mz(M200,z_arr,200.)
        #print dn_dzdm      

        N_z = np.zeros(len(z_arr) - 1)
        for i in xrange (len(z_arr) - 1):
            N_z[i] = np.trapz(dn_dzdm[:,i],M200[:,i+1],np.diff(M200[:,i+1]))

        return N_z

    def N_of_z_SZ(self,z_arr,beams,noises,freqs,clusterDict,lknee,alpha,fileFunc=None,quick=True,tmaxN=5,numts=1000):
        # this is dn/dV(z)

        lnYmin = np.log(1e-13)
        dlnY = 0.1
        lnY = np.arange(lnYmin,lnYmin+10.,dlnY)
    
        Mexp = np.arange(12.5, 15.71, .1)
        #Mexp = np.arange(14.0, 15.4, 0.2)
        rho_crit0m = self.cc.rhoc0om
        hh = self.cc.H0/100

        M = 10.**Mexp
        dM = np.gradient(M)

        M200 = np.outer(M,np.zeros([len(z_arr)]))
        #dM200 = np.outer(M[1:],np.zeros([len(z_arr)]))
        P_func = np.outer(M,np.zeros([len(z_arr)]))
        sigN = np.outer(M,np.zeros([len(z_arr)]))
        M_arr =  np.outer(M,np.ones([len(z_arr)]))


        DA_z = self.cc.results.angular_diameter_distance(z_arr) * hh

        SZProf = SZ_Cluster_Model(self.cc,clusterDict,rms_noises = noises,fwhms=beams,freqs=freqs,lknee=lknee,alpha=alpha)

        for ii in xrange (len(z_arr)-1):
            i = ii + 1
            M200[:,i] = self.cc.Mass_con_del_2_del_mean200(M,500,z_arr[i])
            #dM200[:,i] = np.gradient(M200[:,i])
            #dM200[:,i] = np.diff(M200[:,i])
            for j in xrange(len(M)):
                try:
                    assert fileFunc is not None
                    filename = fileFunc(Mexp[j],z_arr[i])
                    print filename
                    sigN[j,i] = np.loadtxt(filename,unpack=True)[-1]
                except:
                    #print "Calculating S/N because file not found or specified for M=",Mexp[j]," z=",z_arr[i]
                    if quick:
                        var = SZProf.quickVar(M[j],z_arr[i],tmaxN,numts)
                    else:
                        var = SZProf.filter_variance(M[j],z_arr[i])
                    sigN[j,i] = np.sqrt(var)
                #print Mexp[j],z_arr[i]
            
            P_func[:,i] = SZProf.P_of_q (lnY,M_arr[:,i],z_arr[i],sigN[:,i])#*dlnY

        #print P_func
        #dN_dzdm = self.N_of_Mz(M200,z_arr,200.)
        dn_dzdm = self.dn_dM(M200,z_arr,200.)
        #print dn_dzdm
        N_z = np.zeros(len(z_arr) - 1)
        for i in xrange (len(z_arr) - 1):
            N_z[i] = np.trapz(dn_dzdm[:,i+1]*P_func[:,i+1],M200[:,i+1],np.diff(M200[:,i+1]))

        #N_z = np.dot(dN_dzdm,np.transpose(P_func[:,1:]*dM200[:,1:]))

        return N_z

    def N_of_mz_SZ(self,z_arr,beams,noises,freqs,clusterDict,lknee,alpha,fileFunc=None,quick=True,tmaxN=5,numts=1000):
        # this is d^2N/dzdm 
        lnYmin = np.log(1e-13)
        dlnY = 0.1
        lnY = np.arange(lnYmin,lnYmin+10.,dlnY)

        Mexp = np.arange(13.5, 15.0, .1)

        M = 10.**Mexp
        dM = np.gradient(M)
        rho_crit0m = self.cc.rhoc0om
        hh = self.cc.H0/100.

        M200 = np.outer(M,np.zeros([len(z_arr)]))
        dM200 = np.outer(M,np.zeros([len(z_arr)]))
        P_func = np.outer(M,np.zeros([len(z_arr)]))
        sigN = np.outer(M,np.zeros([len(z_arr)]))
        M_arr =  np.outer(M,np.ones([len(z_arr)]))
        dN_dzdm =  np.outer(M,np.ones([len(z_arr)-1]))

        DA_z = self.cc.results.angular_diameter_distance(z_arr) * hh

        SZProf = SZ_Cluster_Model(self.cc,clusterDict,rms_noises = noises,fwhms=beams,freqs=freqs,lknee=lknee,alpha=alpha)

        for ii in xrange (len(z_arr)-1):
            i = ii + 1
            M200[:,i] = self.cc.Mass_con_del_2_del_mean200(M,500,z_arr[i])
            dM200[:,i] = np.gradient(M200[:,i])
            for j in xrange(len(M)):
                try:
                    assert fileFunc is not None
                    filename = fileFunc(Mexp[j],z_arr[i])
                    print filename
                    sigN[j,i] = np.loadtxt(filename,unpack=True)[-1]
                except:
                    #print "Calculating S/N because file not found or specified for M=",Mexp[j]," z=",z_arr[i]
                    if quick:
                        var = SZProf.quickVar(M[j],z_arr[i],tmaxN,numts)
                    else:
                        var = SZProf.filter_variance(M[j],z_arr[i])
                    sigN[j,i] = np.sqrt(var)
                #print Mexp[j],z_arr[i]

            P_func[:,i] = SZProf.P_of_q (lnY,M_arr[:,i],z_arr[i],sigN[:,i])*dlnY

        dn_dVdm = self.dn_dM(M200,z_arr,200.)
        dV_dz = self.dVdz(z_arr)
        dN_dzdm = dn_dVdm[:,1:]*P_func[:,1:]* dV_dz[1:]
        
        return dN_dzdm,dM200[:,0]

    def Mass_err (self,mass_err,z_arr,beams,noises,freqs,clusterDict,lknee,alpha,fileFunc=None,quick=True,tmaxN=5,numts=1000):
        # this is TEMP WL MASS ERROR
        alpha_ym = self.cc.paramDict['alpha_ym'] #1.79   
        lnYmin = np.log(1e-13)
        dlnY = 0.1
        lnY = np.arange(lnYmin,lnYmin+10.,dlnY)

        #Mexp = np.arange(13.5, 15.71, .1)
        Mexp = np.arange(12.5,15.5,0.05)+0.05 
        M = 10.**Mexp
        dM = np.gradient(M)
        rho_crit0m = self.cc.rhoc0om
        hh = self.cc.H0/100.
        #hh = 0.7

        M200 = np.outer(M,np.zeros([len(z_arr)]))
        dM200 = np.outer(M,np.zeros([len(z_arr)]))
        P_func = np.outer(M,np.zeros([len(z_arr)]))
        sigN = np.outer(M,np.zeros([len(z_arr)]))
        YM   =  np.outer(M,np.ones([len(z_arr)]))
        M_arr =  np.outer(M,np.ones([len(z_arr)]))

        #HSC_mass = np.loadtxt('input/HSC_DeltalnM_z0_z2.txt',unpack=True)
        #HSC_mass = np.transpose(HSC_mass)

        #CMB_halo = np.loadtxt('data/AdvACTCMBLensingWhiteNoise150GhzTTOnly.dat',unpack=True)
        #CMB_halo *= np.sqrt(1000.)
        #CMB_halo = np.transpose(CMB_halo)

        #mhalo = np.arange(14.0,15.7,0.05)
        #zhalo = np.arange(0.1,2.0,0.05)

        #CMB_mass = interpolateGrid(CMB_halo,mhalo,zhalo,np.log10(M) ,z_arr[1:])

        #mass_err = 1./(np.sqrt((1./HSC_mass)**2 + (1./CMB_mass)**2))

        DA_z = self.cc.results.angular_diameter_distance(z_arr) * hh

        SZProf = SZ_Cluster_Model(self.cc,clusterDict,rms_noises = noises,fwhms=beams,freqs=freqs,lknee=lknee,alpha=alpha)

        for ii in xrange (len(z_arr)-1):
            i = ii + 1
            M200[:,i] = self.cc.Mass_con_del_2_del_mean200(M,500,z_arr[i])
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
                    YM[j,i] = SZProf.Y_M(M[j],z_arr[i])
                        
                print Mexp[j],z_arr[i]

            P_func[:,i] = SZProf.P_of_q (lnY,M_arr[:,i],z_arr[i],sigN[:,i])*dlnY

        dn_dVdm = self.dn_dM(M200,z_arr,200.)
        dV_dz = self.dVdz(z_arr)

        N_z = np.zeros(len(z_arr) - 1)
        N_tot_z = np.zeros(len(z_arr) - 1)
        for i in xrange (len(z_arr) - 1):
            N_z[i] = np.sum(dn_dVdm[:,i+1]*P_func[:,i+1]*dM200[:,i+1] / (mass_err[:,i]**2 + alpha_ym**2 * (sigN[:,i+1]/YM[:,i+1])**2))
            N_tot_z[i] = np.sum(dn_dVdm[:,i+1]*P_func[:,i+1]*dM200[:,i+1])
        err_WL_mass = 4.*np.pi* (1400./42000.)*np.sum(N_z*dV_dz[1:])*0.05
        Ntot = 4.*np.pi* (1400./42000.)*np.sum(N_tot_z*dV_dz[1:])*0.05

        return 1./err_WL_mass,Ntot

    def N_of_mqz_SZ (self,mass_err,z_arr,m_wl,q_arr,beams,noises,freqs,clusterDict,lknee,alpha,fileFunc=None,quick=True,tmaxN=5,numts=1000):
        # this is 3D grid for fisher matrix 
        lnYmin = np.log(1e-14)
        dlnY = 0.1
        lnY = np.arange(lnYmin,lnYmin+13.,dlnY)
        
        #Mexp = np.arange(13.5, 15.71, .1)
        Mexp = m_wl
        
        M = 10.**Mexp
        dM = np.gradient(M)
        rho_crit0m = self.cc.rhoc0om
        hh = self.cc.H0/100.

        M200 = np.outer(M,np.zeros([len(z_arr)]))
        #dM200 = np.outer(M,np.zeros([len(z_arr)]))
        sigN = np.outer(M,np.zeros([len(z_arr)]))
        YM   =  np.outer(M,np.ones([len(z_arr)]))
        M_arr =  np.outer(M,np.ones([len(z_arr)]))
        #P_func_test = np.outer(M,np.zeros([len(z_arr)]))
        P_func = np.zeros([len(M),len(z_arr)-1,len(q_arr)])
        dNdzmq = np.zeros([len(m_wl),len(z_arr)-1,len(q_arr)])
        dNdzmq2 = np.zeros([len(m_wl),len(z_arr)-1,len(q_arr)])

        # HSC_mass = np.loadtxt('input/HSC_DeltalnM_z0_z2.txt',unpack=True)
        # HSC_mass = np.transpose(HSC_mass)

        # mass_err = HSC_mass#1./(np.sqrt((1./HSC_mass)**2 + (1./CMB_mass)**2))
        
        DA_z = self.cc.results.angular_diameter_distance(z_arr) * hh
        
        #print self.cc.paramDict['b_ym']

        SZProf = SZ_Cluster_Model(self.cc,clusterDict,rms_noises = noises,fwhms=beams,freqs=freqs,lknee=lknee,alpha=alpha)
        
#        for i in xrange (4):
#            for kk in xrange(4):
#                for jj in xrange(4):
#                    print SZProf.Mwl_prob(np.exp(m_wl[jj]),M_arr[:,i+1],mass_err[:,i])


        for ii in xrange (len(z_arr)-1):
            i = ii + 1
            M200[:,i] = self.cc.Mass_con_del_2_del_mean200(M,500,z_arr[i])
            #dM200[:,i] = np.gradient(M200[:,i])
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
                    YM[j,i] = SZProf.Y_M(M[j],z_arr[i])

                print Mexp[j],z_arr[i]
            #P_func_test[:,i] = SZProf.P_of_q (lnY,M_arr[:,i],z_arr[i],sigN[:,i])*dlnY
            for kk in xrange(len(q_arr)):
                P_func[:,ii,kk] = SZProf.P_of_qn (lnY,M_arr[:,i],z_arr[i],sigN[:,i],q_arr[kk])

            #print "PFUNC", np.max(P_func[:,i,:]),np.max(P_func_test[:,i])
            #print "PFUNC", P_func[:,i,0]

        dn_dVdm = self.dn_dM(M200,z_arr,200.)
        dV_dz = self.dVdz(z_arr)

        for kk in xrange(len(q_arr)):
            for jj in xrange(len(m_wl)):
                for i in xrange (len(z_arr) - 1):
                    dNdzmq[jj,i,kk] = np.trapz(dn_dVdm[:,i+1]*P_func[:,i,kk]*SZProf.Mwl_prob(10**(m_wl[jj]),M_arr[:,i+1],mass_err[:,i]),M200[:,i+1],np.diff(M200[:,i+1]))
                    dNdzmq[jj,i,kk] *= dV_dz[i+1]*4.*np.pi
                    #print np.max(SZProf.Mwl_prob(np.exp(m_wl[jj]),M_arr[:,i+1],mass_err[:,i])*P_func[:,i+1,kk]),dNdzmq[jj,i,kk],np.sum(dn_dVdm[:,i+1]*dM200[:,i+1]),dV_dz[i+1]
                #dNdzmq[jj,:,kk] *= dV_dz[1:] * 0.05 * 4.*np.pi* (1400./42000.)
                #print dNdzmq[jj,:,kk], dV_dz[1:]
                #dNdzmq2[jj,:,kk] *= dV_dz[1:] * 0.05 * 4.*np.pi

        return dNdzmq

class SZ_Cluster_Model:
    def __init__(self,clusterCosmology,clusterDict,fwhms=[1.5],rms_noises =[1.], freqs = [150.],lmax=8000,lknee=0.,alpha=1.,dell=10,pmaxN=5,numps=1000,nMax=1,**options):
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
        self.evalells = np.arange(2,lmax,self.dell)
        for freq,fwhm,noise in zip(freqs,fwhms,rms_noises):
            freq_fac = (f_nu(self.cc,freq))**2


            nells = self.cc.clttfunc(self.evalells)+( self.noise_func(self.evalells,fwhm,noise,lknee,alpha) / self.cc.c['TCMBmuK']**2.)
            self.nlinv += (freq_fac)/nells

        self.nl = 1./self.nlinv

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



    
    def Y_erf(self,Y_true,sigma_N):
        q = 5.
        sigma_Na = np.outer(sigma_N,np.ones(len(Y_true[0,:])))
        
        ans = 0.5 * (1. + special.erf((Y_true - q*sigma_Na)/(np.sqrt(2.)*sigma_Na)))
        return ans
    
    def P_of_Y (self,lnY,MM,zz):
        Ysig = self.scaling['Ysig'] #0.127  

        Y = np.exp(lnY)
        Ma = np.outer(MM,np.ones(len(Y[0,:])))
        numer = -1.*(np.log(Y/self.Y_M(Ma,zz)))**2
        ans = 1./(Ysig * np.sqrt(2*np.pi)) * np.exp(numer/(2.*Ysig))
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

    def gaussian(self,xx, mu, sig):
        return 1./(sig * np.sqrt(2*np.pi)) * np.exp(-1.*(xx - mu)**2 / (2. * sig**2.))

    def q_prob (self,q_arr,lnY,sigma_N):
        #Gaussian error probablity for SZ S/N
        sigma_Na = np.outer(sigma_N,np.ones(len(lnY[0,:])))
        Y = np.exp(lnY)
        ans = self.gaussian(q_arr,Y/sigma_Na,1.)        
        return ans

    def Mwl_prob (self,Mwl,M,Merr):
        #Gaussian error probablity for weak lensing mass 
        ans = self.gaussian(Mwl,M,Merr*M)
        return ans

