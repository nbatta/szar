import numpy as np
from szar.counts import ClusterCosmology,Halo_MF
from szar.szproperties import gaussian
import emcee
from nemo import simsTools
from scipy import special,stats
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
    mmap=img2[0].data
    #return the filter noise map for pixels in the mask map that = 1
    return rmsmap*mmap

def read_clust_cat(fitsfile):
    list = fits.open(fitsfile)
    data = list[1].data
    SNR = data.field('SNR2p4')
    ra = data.field('RADeg')
    dec = data.field('DECDeg')
    z = data.field('z')
    zerr = data.field('zErr')
    Y0 = data.field('y0tilde')
    Y0err = data.field('y0tilde_err')
    ind = np.where(SNR >= 5.6)[0]
    #z = data.field('M500_redshift')
    #Y0 = data.field('M500_fixed_y_c')
    #Y0err = data.field('M500_fixed_err_y_c')
    return ra[ind],dec[ind],z[ind],zerr[ind],Y0[ind],Y0err[ind]

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
        clust_cat = nemoOutputDir + 'E-D56Clusters.fits' #'ACTPol_mjh_cluster_cat.fits'

        self.rms_noise_map  = read_MJH_noisemap(FilterNoiseMapFile,MaskMapFile)
        self.wcs=astWCS.WCS(FilterNoiseMapFile) 
        self.clst_RA,self.clst_DEC,self.clst_z,self.clst_zerr,self.clst_y0,self.clst_y0err = read_clust_cat(clust_cat)
        self.clst_xmapInd,self.clst_ymapInd = self.Find_nearest_pixel_ind(self.clst_RA,self.clst_DEC)

        self.qmin = 5.6
        self.num_noise_bins = 20
        self.area_rads = 987.5/41252.9612
        self.LgY = np.arange(-6,-3,0.05)

        count_temp,bin_edge =np.histogram(np.log10(self.rms_noise_map[self.rms_noise_map>0]),bins=self.num_noise_bins)
        self.frac_of_survey = count_temp*1.0 / np.sum(count_temp)
        self.thresh_bin = 10**((bin_edge[:-1] + bin_edge[1:])/2.)

    def alter_fparams(self,fparams,parlist,parvals):
        for k,parvals in enumerate(parvals):
            fparams[parlist[k]] = parvals
        return fparams

    def Find_nearest_pixel_ind(self,RADeg,DECDeg):
        xx = np.array([])
        yy = np.array([])
        for ra, dec in zip(RADeg,DECDeg):
            x,y = self.wcs.wcs2pix(ra,dec)
            np.append(xx,np.round(x))
            np.append(yy,np.round(y))
        #return [np.round(x),np.round(y)]
        return np.round(xx),np.round(yy)

    def PfuncY(self,YNoise,M,z_arr,param_vals):
        LgY = self.LgY
        
        P_func = np.outer(M,np.zeros([len(z_arr)]))
        M_arr =  np.outer(M,np.ones([len(z_arr)]))

        for i in range(z_arr.size):
            P_func[:,i] = self.P_of_gt_SN(LgY,M_arr[:,i],z_arr[i],YNoise,param_vals)
        return P_func

    def P_Yo(self, LgY, M, z,param_vals):
        #M500c has 1/h factors in it
        Ma = np.outer(M,np.ones(len(LgY[0,:])))
        Ytilde, theta0, Qfilt =simsTools.y0FromLogM500(np.log10(param_vals['massbias']*Ma/(param_vals['H0']/100.)), z, self.tckQFit,sigma_int=param_vals['scat'],B0=param_vals['yslope'])#,tenToA0=YNorm)
        Y = 10**LgY
        numer = -1.*(np.log(Y/Ytilde))**2
        ans = 1./(param_vals['scat'] * np.sqrt(2*np.pi)) * np.exp(numer/(2.*param_vals['scat']**2))
        return ans

    def Y_erf(self,Y,Ynoise):
        qmin = self.qmin  # fixed 
        ans = 0.5 * (1. + special.erf((Y - qmin*Ynoise)/(np.sqrt(2.)*Ynoise)))
        return ans

    def P_of_gt_SN(self,LgY,MM,zz,Ynoise,param_vals):
        Y = 10**LgY
        sig_thresh = np.outer(np.ones(len(MM)),self.Y_erf(Y,Ynoise))
        LgYa = np.outer(np.ones(len(MM)),LgY)
        P_Y = self.P_Yo(LgYa,MM,zz,param_vals)
        ans = np.trapz(P_Y*sig_thresh,LgY,np.diff(LgY),axis=1)
        return ans

    def P_of_Y_per(self,LgY,MM,zz,Y_c,Y_err,param_vals):
        P_Y_sig = np.outer(np.ones(len(MM)),self.Y_prob(Y_c,LgY,Y_err))
        LgYa = np.outer(np.ones(len(MM)),LgY)
        P_Y = self.P_Yo(LgYa,MM,zz,param_vals)
        ans = np.trapz(P_Y*P_Y_sig,LgY,np.diff(LgY),axis=1)
        return ans

    def Y_prob (self,Y_c,LgY,YNoise):
        Y = 10**(LgY)
        ans = gaussian(Y,Y_c,YNoise)
        return ans

    def Pfunc_per(self,MM,zz,Y_c,Y_err,param_vals):
        LgY = self.LgY
        LgYa = np.outer(np.ones(len(MM)),LgY)
        P_Y_sig = self.Y_prob(Y_c,LgY,Y_err)
        P_Y = self.P_Yo(LgYa,MM,zz,param_vals)
        ans = np.trapz(P_Y*P_Y_sig,LgY,np.diff(LgY),axis=1)
        return ans

    def Pfunc_per_zarr(self,MM,z_arr,Y_c,Y_err,int_HMF,param_vals):
        LgY = self.LgY

        P_func = np.outer(MM,np.zeros([len(z_arr)]))
        M_arr =  np.outer(MM,np.ones([len(z_arr)]))
        M200 = np.outer(MM,np.zeros([len(z_arr)]))
        for i in range(z_arr.size):
            P_func[:,i] = self.P_of_Y_per(LgY,M_arr[:,i],z_arr[i],Y_c,Y_err,param_vals)
            M200[:,i] = int_HMF.cc.Mass_con_del_2_del_mean200(self.HMF.M.copy(),500,z_arr[i])
        return P_func,M200

    def Ntot_survey(self,int_HMF,fsky,Ythresh,param_vals):

        z_arr = self.HMF.zarr.copy()        
        Pfunc = self.PfuncY(Ythresh,self.HMF.M.copy(),z_arr,param_vals)
        dn_dzdm = int_HMF.dn_dM(int_HMF.M200,200.)

        N_z = np.trapz(dn_dzdm*Pfunc,dx=np.diff(int_HMF.M200,axis=0),axis=0)
        Ntot = np.trapz(N_z*int_HMF.dVdz,dx=np.diff(z_arr))*4.*np.pi*fsky
        return Ntot

    def Prob_per_cluster(self,int_HMF,cluster_props,dn_dzdm_int,param_vals):
        c_z, c_zerr, c_y, c_yerr = cluster_props
        if (c_zerr > 0):
            z_arr = np.arange(-3.*c_zerr,(3.+0.1)*c_zerr,c_zerr) + c_z
            Pfunc_ind,M200 = self.Pfunc_per_zarr(self.HMF.M.copy(),z_arr,c_y,c_yerr,int_HMF,param_vals)
            dn_dzdm = dn_dzdm_int(z_arr,self.HMF.M.copy())
            N_z_ind = np.trapz(dn_dzdm*Pfunc_ind,dx=np.diff(M200,axis=0),axis=0)
            N_per = np.trapz(N_z_ind*gaussian(z_arr,c_z,c_zerr),dx=np.diff(z_arr))
            ans = N_per            
        else:
            Pfunc_ind = self.Pfunc_per(self.HMF.M.copy(),c_z, c_y, c_yerr,param_vals)
            dn_dzdm = dn_dzdm_int(c_z,self.HMF.M.copy())[:,0]
            M200 = int_HMF.cc.Mass_con_del_2_del_mean200(self.HMF.M.copy(),500,c_z)
            N_z_ind = np.trapz(dn_dzdm*Pfunc_ind,dx=np.diff(M200,axis=0),axis=0)
            ans = N_z_ind
        return ans

    def lnprior(self,theta,parlist,priorval,priorlist):
        param_vals = self.alter_fparams(self.fparams,parlist,theta)
        prioravg = priorval[0,:]
        priorwth = priorval[1,:]
        lnp = 0.
        for k,prioravg in enumerate(prioravg):
            lnp += np.log(gaussian(param_vals[priorlist[k]],prioravg,priorwth[k],noNorm=True))
 
        if ((param_vals['scat'] < 0) or (param_vals['tau'] < 0)) :
            lnp += -np.inf 

        return lnp


    def lnlike(self,theta,parlist):
        
        param_vals = self.alter_fparams(self.fparams,parlist,theta)
        int_cc = ClusterCosmology(param_vals,self.constDict,clTTFixFile=self.clttfile) # internal HMF call
        int_HMF = Halo_MF(int_cc,self.mgrid,self.zgrid) # internal HMF call
        dndm_int = int_HMF.inter_dndm(200.) # delta = 200
        cluster_prop = np.array([self.clst_z,self.clst_zerr,self.clst_y0*1e-4,self.clst_y0err*1e-4])

        Ntot = 0.
        for i in range(len(self.frac_of_survey)):
             Ntot += self.Ntot_survey(int_HMF,self.area_rads*self.frac_of_survey[i],self.thresh_bin[i],param_vals)

        Nind = 0
        for i in xrange(len(self.clst_z)):
            N_per = self.Prob_per_cluster(int_HMF,cluster_prop[:,i],dndm_int,param_vals)
            Nind = Nind + np.log(N_per) 
        return -Ntot + Nind

    def lnprob(self,theta, parlist, priorval, priorlist):
        lp = self.lnprior(theta, parlist, priorval, priorlist)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(theta, parlist)

#Functions from NEMO
#y0FromLogM500(log10M500, z, tckQFit, tenToA0 = 4.95e-5, B0 = 0.08, Mpivot = 3e14, sigma_int = 0.2)
#fitQ(parDict, diagnosticsDir, filteredMapsDir)

    #self.diagnosticsDir=nemoOutputDir+os.path.sep+"diagnostics"
    
    #filteredMapsDir=nemoOutputDir+os.path.sep+"filteredMaps"
    #self.tckQFit=simsTools.fitQ(parDict, self.diagnosticsDir, filteredMapsDir)
