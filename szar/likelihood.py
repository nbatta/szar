import numpy as np
from szar.counts import ClusterCosmology,Halo_MF
import emcee
from nemo import simsTools
from scipy import special
from astropy.io import fits
from astLib import astWCS
from configparser import SafeConfigParser
from orphics.tools.io import dictFromSection
import cPickle as pickle
import matplotlib.pyplot as plt

#import time

def read_MJH_noisemap(noiseMap,maskMap):
    #Read in filter noise map
    img = fits.open(noiseMap)
    rmsmap=img[0].data
    #Read in mask map
    img2 = fits.open(maskMap)
    mmap=img[0].data
    #return the filter noise map for pixels in the mask map that = 1
    return rmsmap*mmap

class clusterLike:
    def __init__(self,iniFile,expName,gridName,parDict,nemoOutputDir,noiseFile):
        
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
        self.constDict = dictFromSection(Config,'constants')
        version = Config.get('general','version')
        
        self.mgrid,self.zgrid,siggrid = pickle.load(open(bigDataDir+"szgrid_"+expName+"_"+gridName+ "_v" + version+".pkl",'rb'))
        self.cc = ClusterCosmology(self.fparams,self.constDict,clTTFixFile=self.clttfile)
        self.HMF = Halo_MF(self.cc,self.mgrid,self.zgrid)

        self.diagnosticsDir=nemoOutputDir+"diagnostics" 
        self.filteredMapsDir=nemoOutputDir+"filteredMaps"
        self.tckQFit=simsTools.fitQ(parDict, self.diagnosticsDir, self.filteredMapsDir)
        FilterNoiseMapFile = nemoOutputDir + noiseFile
        MaskMapFile = self.diagnosticsDir + '/areaMask.fits'
        self.rms_noise_map  = read_MJH_noisemap(FilterNoiseMapFile,MaskMapFile)
        self.wcs=astWCS.WCS(FilterNoiseMapFile) 
        self.qmin = 5.6
        self.Ysig = 0.2
        self.LgY = np.arange(-6,-3,0.05)

    def Find_nearest_pixel_ind(self,RADeg,DECDeg):
        x,y = self.wcs.wcs2pix(RADeg,DECDeg)
        return [np.round(x),np.round(y)]

    def PfuncY(self,YNoise,M,z_arr):
        LgY = self.LgY
        
        P_func = np.outer(M,np.zeros([len(z_arr)]))
        M_arr =  np.outer(M,np.ones([len(z_arr)]))

        for i in range(z_arr.size):
            P_func[:,i] = self.P_of_SN(LgY,M_arr[:,i],z_arr[i],YNoise)
        return P_func

    def P_Yo(self, LgY, M, z):#,thetaScalPars):
        #YNorm,Yslope,Ysig = thetaScalPars
        Ma = np.outer(M,np.ones(len(LgY[0,:])))
        Ytilde, theta0, Qfilt =simsTools.y0FromLogM500(np.log10(Ma), z, self.tckQFit)#,tenToA0=YNorm,B0=YSlope,sigma_int=Ysig)

        Y = 10**LgY
        numer = -1.*(np.log(Y/Ytilde))**2
        ans = 1./(self.Ysig * np.sqrt(2*np.pi)) * np.exp(numer/(2.*self.Ysig**2))
        
        return ans

    def Y_erf(self,Y,Ynoise):
        qmin = self.qmin  # fixed 
        ans = 0.5 * (1. + special.erf((Y - qmin*Ynoise)/(np.sqrt(2.)*Ynoise)))
        return ans

    def P_of_SN(self,LgY,MM,zz,Ynoise):#,thetaScalPars):
        Y = 10**LgY
        sig_thresh = np.outer(np.ones(len(MM)),self.Y_erf(Y,Ynoise))
        Ya = np.outer(np.ones(len(MM)),LgY)
        P_Y = self.P_Yo(Ya,MM,zz)#,thetaScalPars)
        ans = np.trapz(P_Y*sig_thresh,LgY,np.diff(LgY),axis=1)
        return ans
    
    def q_prob (self,LgY,YNoise):
        #Gaussian error probablity for SZ S/N                                                                                 
        YNoise = np.outer(sigma_N,np.ones(len(LgY[0,:])))
        Y = 10**(lgY)
        ans = gaussian(q,Y/YNoise,1.)
        return ans

    def Ntot_survey(self,fsky,Ythresh):
        #temp
        #Ythresh = 10**(-4.65)

        z_arr = self.HMF.zarr.copy()
        
        Pfunc = self.PfuncY(Ythresh,self.HMF.M.copy(),z_arr)
        dn_dzdm = self.HMF.dn_dM(self.HMF.M200,200.)

        print ((dn_dzdm*Pfunc).shape,(np.diff(self.HMF.M200,axis=0).shape))
        N_z = np.trapz(dn_dzdm*Pfunc,dx=np.diff(self.HMF.M200,axis=0),axis=0)
        Ntot = np.trapz(N_z*self.HMF.dVdz,dx=np.diff(z_arr))*4.*np.pi*fsky
        return Ntot

    def lnprior(self,theta):
        a1,a2 = theta
        if  1 < a1 < 5 and  1 < a2 < 2:
            return 0
        return -np.inf

    def lnlike(self,fparams,clustsz,nemo):
        self.cc = ClusterCosmology(fparams,self.constDict,clTTFixFile=self.clttfile)
        self.HMF = Halo_MF(cc,self.mgrid,self.zgrid)

        Ntot = 1.
        Nind = 0
        for i in xrange(len(clustsz)):
            N_per = 1.
            Nind = Nind + np.log(N_per) 
        return -Ntot * Nind

    def lnprob(self,theta, inter, mthresh, zthresh):
        lp = self.lnprior(theta, mthresh, zthresh)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(theta, inter)

#Functions from NEMO
#y0FromLogM500(log10M500, z, tckQFit, tenToA0 = 4.95e-5, B0 = 0.08, Mpivot = 3e14, sigma_int = 0.2)
#fitQ(parDict, diagnosticsDir, filteredMapsDir)

    #self.diagnosticsDir=nemoOutputDir+os.path.sep+"diagnostics"
    
    #filteredMapsDir=nemoOutputDir+os.path.sep+"filteredMaps"
    #self.tckQFit=simsTools.fitQ(parDict, self.diagnosticsDir, filteredMapsDir)
