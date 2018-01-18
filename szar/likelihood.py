import numpy as np
from szar.counts import ClusterCosmology,Halo_MF
from szar.szproperties import gaussian
import emcee
import simsTools
from scipy import special,stats
from astropy.io import fits
from astLib import astWCS
from configparser import SafeConfigParser
from orphics.io import dict_from_section
import cPickle as pickle
import matplotlib.pyplot as plt

#import time
from enlib import bench

def read_MJH_noisemap(noiseMap,maskMap):
    #Read in filter noise map
    img = fits.open(noiseMap)
    rmsmap=img[0].data
    #Read in mask map
    img2 = fits.open(maskMap)
    mmap=img2[0].data
    return rmsmap*mmap

def read_clust_cat(fitsfile):
    list = fits.open(fitsfile)
    data = list[1].data
    SNR = data.field('SNR2p4')
    z = data.field('z')
    zerr = data.field('zErr')
    Y0 = data.field('y0tilde')
    Y0err = data.field('y0tilde_err')
    ind = np.where(SNR >= 5.6)[0]
    return z[ind],zerr[ind],Y0[ind],Y0err[ind]

def read_mock_cat(fitsfile):
    list = fits.open(fitsfile)
    data = list[1].data
    SNR = data.field('fixed_SNR')
    z = data.field('redshift')
    zerr = data.field('redshiftErr')
    Y0 = data.field('fixed_y_c')
    Y0err = data.field('err_fixed_y_c')
    ind = np.where(SNR >= 5.6)[0]
    return z[ind],zerr[ind],Y0[ind],Y0err[ind]

class clusterLike:
    def __init__(self,iniFile,parDict,nemoOutputDir,noiseFile,fix_params,test=False,simtest=False,simpars=False):
        self.fix_params = fix_params
        self.test = test
        self.simtest = simtest
        self.simpars = simpars
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
        self.constDict = dict_from_section(Config,'constants')
        version = Config.get('general','version')
        
        #self.mgrid,self.zgrid,siggrid = pickle.load(open(bigDataDir+"szgrid_"+expName+"_"+gridName+ "_v" + version+".pkl",'rb'))
        logm_min = 12.7
        logm_max = 15.72
        logm_spacing = 0.04
        self.mgrid = np.arange(logm_min,logm_max,logm_spacing)
        self.zgrid = np.arange(0.0,2.01,0.1)        
        #print self.mgrid
        #print self.zgrid
        

        self.cc = ClusterCosmology(self.fparams,self.constDict,clTTFixFile=self.clttfile)
        self.HMF = Halo_MF(self.cc,self.mgrid,self.zgrid)

        self.diagnosticsDir=nemoOutputDir+"diagnostics" 
        self.filteredMapsDir=nemoOutputDir+"filteredMaps"
        self.tckQFit=simsTools.fitQ(parDict, self.diagnosticsDir, self.filteredMapsDir)
        FilterNoiseMapFile = nemoOutputDir + noiseFile
        MaskMapFile = self.diagnosticsDir + '/areaMask.fits'
        
        if self.simtest or self.simpars:
            clust_cat = nemoOutputDir + 'mockCatalog_equD56.fits' #'ACTPol_mjh_cluster_cat.fits'
            self.clst_z,self.clst_zerr,self.clst_y0,self.clst_y0err = read_mock_cat(clust_cat)
        else:
            clust_cat = nemoOutputDir + 'E-D56Clusters.fits' #'ACTPol_mjh_cluster_cat.fits'
            self.clst_z,self.clst_zerr,self.clst_y0,self.clst_y0err = read_clust_cat(clust_cat)

        self.rms_noise_map  = read_MJH_noisemap(FilterNoiseMapFile,MaskMapFile)
        #self.wcs=astWCS.WCS(FilterNoiseMapFile) 
        #self.clst_RA,self.clst_DEC,
        #self.clst_xmapInd,self.clst_ymapInd = self.Find_nearest_pixel_ind(self.clst_RA,self.clst_DEC)

        self.qmin = 5.6
        self.num_noise_bins = 10
        self.area_rads = 987.5/41252.9612 # fraction of sky - ACTPol D56-equ specific
        self.LgY = np.arange(-6,-3,0.05)

        count_temp,bin_edge =np.histogram(np.log10(self.rms_noise_map[self.rms_noise_map>0]),bins=self.num_noise_bins)
        self.frac_of_survey = count_temp*1.0 / np.sum(count_temp)
        self.thresh_bin = 10**((bin_edge[:-1] + bin_edge[1:])/2.)

    def alter_fparams(self,fparams,parlist,parvals):
        for k,parvals in enumerate(parvals):
            fparams[parlist[k]] = parvals
        return fparams

    def Find_nearest_pixel_ind(self,wcs,RADeg,DECDeg):
        xx = np.array([])
        yy = np.array([])
        for ra, dec in zip(RADeg,DECDeg):
            x,y = wcs.wcs2pix(ra,dec)
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
        Om = (param_vals['omch2'] + param_vals['ombh2']) / (param_vals['H0']/100.)**2
        Ytilde, theta0, Qfilt =simsTools.y0FromLogM500(np.log10(param_vals['massbias']*Ma/(param_vals['H0']/100.)), z, self.tckQFit,sigma_int=param_vals['scat'],B0=param_vals['yslope'], H0 = param_vals['H0'], OmegaM0 = Om, OmegaL0 = 1 - Om)#,tenToA0=YNorm)
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
        P_Y = np.nan_to_num(self.P_Yo(LgYa,MM,zz,param_vals))
        ans = np.trapz(P_Y*sig_thresh,LgY,np.diff(LgY),axis=1)
        return ans

    def P_of_Y_per(self,LgY,MM,zz,Y_c,Y_err,param_vals):
        P_Y_sig = np.outer(np.ones(len(MM)),self.Y_prob(Y_c,LgY,Y_err))
        LgYa = np.outer(np.ones(len(MM)),LgY)
        P_Y = np.nan_to_num(self.P_Yo(LgYa,MM,zz,param_vals))
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
        P_Y = np.nan_to_num(self.P_Yo(LgYa,MM,zz,param_vals))
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
        #print Pfunc
        N_z = np.trapz(dn_dzdm*Pfunc,dx=np.diff(int_HMF.M200,axis=0),axis=0)
        Ntot = np.trapz(N_z*int_HMF.dVdz,dx=np.diff(z_arr))*4.*np.pi*fsky
        return Ntot

    def Prob_per_cluster(self,int_HMF,cluster_props,dn_dzdm_int,param_vals):
        c_z, c_zerr, c_y, c_yerr = cluster_props
        if (c_zerr > 0):
            z_arr = np.arange(-3.*c_zerr,(3.+0.1)*c_zerr,c_zerr) + c_z
            Pfunc_ind,M200 = self.Pfunc_per_zarr(int_HMF.M.copy(),z_arr,c_y,c_yerr,int_HMF,param_vals)
            dn_dzdm = dn_dzdm_int(z_arr,np.log10(int_HMF.M.copy()))
            N_z_ind = np.trapz(dn_dzdm*Pfunc_ind,dx=np.diff(M200,axis=0),axis=0)
            N_per = np.trapz(N_z_ind*gaussian(z_arr,c_z,c_zerr),dx=np.diff(z_arr))
            ans = N_per            
        else:
            Pfunc_ind = self.Pfunc_per(int_HMF.M.copy(),c_z, c_y, c_yerr,param_vals)
            #print "PFunc",Pfunc_ind
            M200 = int_HMF.cc.Mass_con_del_2_del_mean200(int_HMF.M.copy(),500,c_z)
            dn_dzdm = dn_dzdm_int(c_z,np.log10(int_HMF.M.copy()))[:,0]
            #print "dndm", dn_dzdm
            #print "M200", M200
            N_z_ind = np.trapz(dn_dzdm*Pfunc_ind,dx=np.diff(M200,axis=0),axis=0)
            ans = N_z_ind
        #print Pfunc_ind
        return ans

    def lnprior(self,theta,parlist,priorval,priorlist):
        param_vals = self.alter_fparams(self.fparams,parlist,theta)
        prioravg = priorval[0,:]
        priorwth = priorval[1,:]
        lnp = 0.
        for k,prioravg in enumerate(prioravg):
            lnp += np.log(gaussian(param_vals[priorlist[k]],prioravg,priorwth[k],noNorm=True))

        try:
            if ((param_vals['scat'] < 0) or (param_vals['tau'] < 0)) :
                lnp += -np.inf
        except:
            pass
        
        if (self.simpars):
            pars = ['omch2','ombh2','H0','As','ns']
            mins = [0.001,0.005,40.,7.389056098930651e-10,0.8]
            maxs = [0.99,0.1,100.,5.459815003314424e-09,1.2]
        else:
            pars = ['omch2','ombh2','H0','As','ns','massbias','yslope','scat']
            mins = [0.001,0.005,40.,7.389056098930651e-10,0.8,0.2,-0.6,0.001 ]
            maxs = [0.99,0.1,100.,5.459815003314424e-09,1.2,1.4,0.6,0.8 ]

        for k,par in enumerate(pars):
            if param_vals[par]<mins[k] or param_vals[par]>maxs[k]: lnp += -np.inf

        return lnp


    def lnlike(self,theta,parlist):

        
        param_vals = self.alter_fparams(self.fparams,parlist,theta)
        for key in self.fix_params:
            if key not in param_vals.keys(): param_vals[key] = self.fix_params[key]

        int_cc = ClusterCosmology(param_vals,self.constDict,clTTFixFile=self.clttfile) # internal HMF call
        int_HMF = Halo_MF(int_cc,self.mgrid,self.zgrid) # internal HMF call
        self.s8 = int_HMF.cc.s8

        if np.nan_to_num(self.s8)<0.1 or np.nan_to_num(self.s8)>10. or not(np.isfinite(self.s8)):
            self.s8 = 0.
        #     return -np.inf
        
        #dndm_int = int_HMF.inter_dndm(200.) # delta = 200
        dndm_int = int_HMF.inter_dndmLogm(200.) # delta = 200
        cluster_prop = np.array([self.clst_z,self.clst_zerr,self.clst_y0*1e-4,self.clst_y0err*1e-4])

        if self.test:
            Ntot = 60.
        else:
            Ntot = 0.
            for i in range(len(self.frac_of_survey)):
                Ntot += self.Ntot_survey(int_HMF,self.area_rads*self.frac_of_survey[i],self.thresh_bin[i],param_vals)
        #print 'NTOT', Ntot
        Nind = 0
        for i in xrange(len(self.clst_z)):
            
            N_per = self.Prob_per_cluster(int_HMF,cluster_prop[:,i],dndm_int,param_vals)
            #if (i < 3):
                #print np.log(N_per)
            Nind = Nind + np.log(N_per)
            #print N_per
        #print Nind
        return -Ntot + Nind

    def lnprob(self,theta, parlist, priorval, priorlist):
        if not (self.simtest):
            lp = self.lnprior(theta, parlist, priorval, priorlist)
            if not np.isfinite(lp):
                return -np.inf, 0.
        else:
            lp = 0
        lnlike = self.lnlike(theta, parlist)
        return lp + lnlike,np.nan_to_num(self.s8)

class MockCatalog:
    def __init__(self,iniFile,parDict,nemoOutputDir,noiseFile,mass_grid_log=None,z_grid=None):

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
        self.constDict = dict_from_section(Config,'constants')
        version = Config.get('general','version')

        if mass_grid_log:
            logm_min,logm_max,logm_spacing = mass_grid_log
        else:
            logm_min = 12.7
            logm_max = 15.72
            logm_spacing = 0.04
        if z_grid:
            zmin,zmax,zdel = z_grid
        else:
            zmin = 0.0
            zmax = 2.01
            zdel = 0.1
        
        self.mgrid = np.arange(logm_min,logm_max,logm_spacing)
        self.zgrid = np.arange(zmin,zmax,zdel)

        self.cc = ClusterCosmology(self.fparams,self.constDict,clTTFixFile=self.clttfile)
        self.HMF = Halo_MF(self.cc,self.mgrid,self.zgrid)

        self.diagnosticsDir=nemoOutputDir+"diagnostics"
        self.filteredMapsDir=nemoOutputDir+"filteredMaps"
        self.tckQFit=simsTools.fitQ(parDict, self.diagnosticsDir, self.filteredMapsDir)
        FilterNoiseMapFile = nemoOutputDir + noiseFile
        MaskMapFile = self.diagnosticsDir + '/areaMask.fits'

        self.rms_noise_map = read_MJH_noisemap(FilterNoiseMapFile,MaskMapFile)
        
        self.fsky = 987.5/41252.9612 # in rads ACTPol D56-equ specific
        self.scat_val = 0.2
        self.seedval = 1

    def Total_clusters(self,fsky):
        Nz = self.HMF.N_of_z()
        ans = np.trapz(Nz*fsky,dx=np.diff(self.HMF.zarr))
        return ans

    def create_basic_sample(self,fsky):
        # create simple mock catalog of Mass and Redshift 
        Ntot100 = np.ceil(self.Total_clusters(fsky) / 100.)
        mlim = [np.min(self.mgrid),np.max(self.mgrid)]
        zlim = [np.min(self.zgrid),np.max(self.zgrid)]
        #mlim = [np.log10(3e14),np.log10(7e15)]
        samples = self.HMF.mcsample_mf(200.,Ntot100,mthresh = mlim,zthresh = zlim)
        
        return samples[:,0],samples[:,1]

    def plot_basic_sample(self):
        fsky = self.fsky
        sampZ,sampM = self.create_basic_sample(fsky)
        plt.figure()
        plt.plot(sampZ,sampM,'x') 
        plt.savefig('default_mockcat.png', bbox_inches='tight',format='png')  
        return

    def create_obs_sample(self,fsky):
        
        #include observational effects like scatter and noise into the detection of clusters
        sampZ,sampM = self.create_basic_sample(fsky)
        nsamps = len(sampM)
        Ytilde = sampM * 0.0
        
        for i in range(nsamps):
            Ytilde[i], theta0, Qfilt =simsTools.y0FromLogM500(np.log10(10**sampM[i]/(self.HMF.cc.H0/100.)), sampZ[i], self.tckQFit)
        # add scatter 4
        np.random.seed(self.seedval)
        ymod = np.exp(self.scat_val * np.random.randn(nsamps))
        sampY0 = Ytilde*ymod
        
        #calculate noise for a given object for a random place on the map and save coordinates

        np.random.seed(self.seedval+1)
        nmap = self.rms_noise_map[::-1,:]
        
        ylims = nmap.shape[0]
        xlims = nmap.shape[1]
        xsave = np.array([])
        ysave = np.array([])
        sampY0err = np.array([])
        count = 0

        while count < nsamps:
            ytemp = np.int32(np.floor(np.random.uniform(0,ylims)))
            xtemp = np.int32(np.floor(np.random.uniform(0,xlims)))
            if nmap[ytemp,xtemp] > 0:
                count += 1
                xsave = np.append(xsave,xtemp)
                ysave = np.append(ysave,ytemp)
                sampY0err = np.append(sampY0err,nmap[ytemp,xtemp])
        return xsave,ysave,sampZ,sampY0,sampY0err,sampY0/sampY0err,sampM


    def plot_obs_sample(self,filename1='default_mockobscat.png',filename2='default_obs_mock_footprint.png'):
        fsky = self.fsky
        xsave,ysave,sampZ,sampY0,sampY0err,SNR,sampM = self.create_obs_sample(fsky)
        ind = np.where(SNR >= 5.6)[0]
        plt.figure()
        plt.plot(sampZ,sampM,'x')
        plt.plot(sampZ[ind],sampM[ind],'o')
        plt.savefig(filename1, bbox_inches='tight',format='png')

        nmap = self.rms_noise_map[::-1,:]
        plt.figure(figsize=(40,6))
        plt.imshow(nmap,cmap='Blues')
        plt.plot(xsave[ind],ysave[ind],'ko')
        plt.colorbar()
        plt.savefig(filename2, bbox_inches='tight',format='png')

        return xsave,ysave,sampZ,sampY0,sampY0err,SNR,sampM

    def write_obs_cat_toFits(self, filename):
        #fsky = self.fsky
        #xsave,ysave,sampZ,sampY0,sampY0err,SNR,sampM = self.create_obs_sample(fsky)

        xsave,ysave,sampZ,sampY0,sampY0err,SNR,sampM = self.plot_obs_sample()

        hdu = fits.BinTableHDU.from_columns(
            [fits.Column(name='Cluster_ID', format='20A', array=a1),
             fits.Column(name='x_ind', format='E', array=xsave),
             fits.Column(name='y_ind', format='E', array=ysave),
             fits.Column(name='redshift', format='E', array=sampZ),
             fits.Column(name='redshiftErr', format='E', array=sampZ*0.0),
             fits.Column(name='fixed_y_c', format='E', array=sampY0),
             fits.Column(name='err_fixed_y_c', format='E', array=sampY0err),
             fits.Column(name='fixed_SNR', format='E', array=SNR),])

        hdu.writeto(filename)

        return 0
