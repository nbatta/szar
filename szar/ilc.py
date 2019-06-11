from __future__ import print_function
from __future__ import division
from builtins import str
from builtins import range
from builtins import object
from past.utils import old_div
import numpy as np
from scipy.interpolate import interp1d
from orphics.cosmology import noise_func
from orphics.cosmology import LensForecast
from szar.foregrounds import fgNoises, f_nu
from orphics.io import Plotter
import numpy.matlib
from scipy.special import j1


def weightcalculator(f,N):
    N_i=np.linalg.inv(N)
    C=np.matmul(np.transpose(f),np.matmul(N_i,f))
    W=(1/C)*np.matmul(np.transpose(f),N_i)
    return W

def constweightcalculator(f_1,f_2,N):
    C=np.matmul(np.transpose(f_1),np.matmul(N,f_1))*np.matmul(np.transpose(f_2),np.matmul(N,f_2))-(np.matmul(np.transpose(f_2),np.matmul(N,f_1)))**2
    M=np.matmul(np.transpose(f_1),np.matmul(N,f_1))*np.matmul(np.transpose(f_2),N)-np.matmul(np.transpose(f_2),np.matmul(N,f_1))*np.matmul(np.transpose(f_1),N)
    W=M/C
    return W

def doubleweightcalculator(f_1,f_2,N):
    C=np.matmul(np.transpose(f_1),np.matmul(N,f_1))*np.matmul(np.transpose(f_2),np.matmul(N,f_2))-(np.matmul(np.transpose(f_2),np.matmul(N,f_1)))**2
    M=(np.matmul(np.transpose(f_1),np.matmul(N,f_1)) - np.matmul(np.transpose(f_2),np.matmul(N,f_1))) *np.matmul(np.transpose(f_2),N) \
        - (np.matmul(np.transpose(f_2),np.matmul(N,f_2)) - np.matmul( np.transpose(f_2),np.matmul(N,f_1)))*np.matmul(np.transpose(f_1),N)
    W=M/C
    return W

class ILC_simple(object):
    def __init__(self,clusterCosmology, fgs,fwhms=[1.5],rms_noises =[1.], freqs = [150.],lmax=8000,lknee=0.,alpha=1.,dell=1.,v3mode=-1,fsky=None,noatm=False):
        
        #Inputs
        #clusterCosmology is a class that contains cosmological parameters and power spectra.
        #fgs is a class that contains the functional forms for the foregrounds and constants

        #Options

        #initial set up for ILC
        self.cc = clusterCosmology

        #initializing the frequency matrices

        self.fgs = fgs

    
        self.dell = dell
        #set-up ells to evaluate up to lmax
        self.evalells = np.arange(2,lmax,self.dell)
        self.N_ll_tsz = self.evalells*0.0
        self.N_ll_cmb = self.evalells*0.0
        self.N_ll_rsx = self.evalells*0.0
        self.N_ll_rsx_NoFG = self.evalells*0.0 
        self.N_ll_cmb_NoFG = self.evalells*0.0 
        self.N_ll_cmb_c_tsz = self.evalells*0.0
        self.N_ll_tsz_c_cmb = self.evalells*0.0
        self.N_ll_tsz_c_cib = self.evalells*0.0

        #Only for SO forecasts, including the SO atmosphere modeling
        if v3mode>-1:
            print("V3 flag enabled.")
            import szar.V3_calc_public as v3
            import szar.so_noise_lat_v3_1_CAND as v3_1

            if v3mode <= 2:
                lat = v3_1.SOLatV3point1(v3mode,el=50.)
                vfreqs = lat.get_bands()# v3.Simons_Observatory_V3_LA_bands()                                                               
                print("Simons Obs")
                print("Replacing ",freqs,  " with ", vfreqs)
                N_bands = len(vfreqs)
                freqs = vfreqs
                vbeams = lat.get_beams()#v3.Simons_Observatory_V3_LA_beams()                                                                
                print("Replacing ",fwhms,  " with ", vbeams)
                fwhms = vbeams

                v3lmax = self.evalells.max()
                v3dell = np.diff(self.evalells)[0]
                print("Using ",fsky," for fsky")

                v3ell,N_ell_T_LA_full, N_ell_P_LA = lat.get_noise_curves(fsky, v3lmax+v3dell, v3dell, full_covar=True, deconv_beam=True)

                N_ell_T_LA = np.diagonal(N_ell_T_LA_full).T
                Map_white_noise_levels = lat.get_white_noise(fsky)**.5

            #if v3mode <= 2:
            #    vfreqs = v3.Simons_Observatory_V3_LA_bands()
            #    freqs = vfreqs
            #    vbeams = v3.Simons_Observatory_V3_LA_beams()
            #    fwhms = vbeams

            #    v3lmax = self.evalells.max()
            #    v3dell = np.diff(self.evalells)[0]

            #    v3ell, N_ell_T_LA, N_ell_P_LA, Map_white_noise_levels = v3.Simons_Observatory_V3_LA_noise(sensitivity_mode=v3mode,f_sky=fsky,ell_max=v3lmax+v3dell,delta_ell=v3dell)
            elif v3mode == 3:
                vfreqs = v3.AdvACT_bands()
                freqs = vfreqs
                vbeams = v3.AdvACT_beams()
                fwhms = vbeams

                v3lmax = self.evalells.max()
                v3dell = np.diff(self.evalells)[0]
                v3ell, N_ell_T_LA, N_ell_P_LA, Map_white_noise_levels = v3.AdvACT_noise(f_sky=fsky,ell_max=v3lmax+v3dell,delta_ell=\
v3dell)
            elif v3mode == 5:
                import szar.lat_noise_190528_w350ds4 as ccatp
                tubes = (0,0,0,2,2,1)
                lat = ccatp.CcatLatv2(v3mode,el=50.,survey_years=4000/24./365.24,survey_efficiency=1.0,N_tubes=tubes)
                vfreqs = lat.get_bands()# v3.Simons_Observatory_V3_LA_bands()
                print("CCATP")
                print("Replacing ",freqs,  " with ", vfreqs)
                N_bands = len(vfreqs)
                freqs = vfreqs
                vbeams = lat.get_beams()#v3.Simons_Observatory_V3_LA_beams() 
                print("Replacing ",fwhms,  " with ", vbeams)
                fwhms = vbeams

                v3lmax = self.evalells.max()
                v3dell = np.diff(self.evalells)[0]
                print("Using ",fsky," for fsky")

                v3ell,N_ell_T_LA_full, N_ell_P_LA = lat.get_noise_curves(fsky, v3lmax+v3dell, v3dell, full_covar=True, deconv_beam=True)

                N_ell_T_LA = np.diagonal(N_ell_T_LA_full).T
                Map_white_noise_levels = lat.get_white_noise(fsky)**.5

            elif v3mode == 6:
                import szar.lat_noise_190528_w350ds4 as ccatp
                #tubes = (0,0,0,2,2,1)
                lat = ccatp.CcatSOLat(v3mode,el=50.,survey_years=4000/24./365.24,survey_efficiency=1.0)
                vfreqs = lat.get_bands()# v3.Simons_Observatory_V3_LA_bands()
                print("CCATP + SO goal")
                print("Replacing ",freqs,  " with ", vfreqs)
                N_bands = len(vfreqs)
                freqs = vfreqs
                vbeams = lat.get_beams()#v3.Simons_Observatory_V3_LA_beams() 
                print("Replacing ",fwhms,  " with ", vbeams)
                fwhms = vbeams

                v3lmax = self.evalells.max()
                v3dell = np.diff(self.evalells)[0]
                print("Using ",fsky," for fsky")

                v3ell,N_ell_T_LA_full, N_ell_P_LA = lat.get_noise_curves(fsky, v3lmax+v3dell, v3dell, full_covar=True, deconv_beam=True)

                N_ell_T_LA = np.diagonal(N_ell_T_LA_full).T
                Map_white_noise_levels = lat.get_white_noise(fsky)**.5

        if (len(freqs) > 1):
            fq_mat   = np.matlib.repmat(freqs,len(freqs),1) 
            fq_mat_t = np.transpose(np.matlib.repmat(freqs,len(freqs),1))
        else:
            fq_mat   = freqs
            fq_mat_t = freqs

        #initializing the weighting functions for the ilc
        #thermal SZ weights
        self.W_ll_tsz = np.zeros([len(self.evalells),len(np.array(freqs))])
        #CMB weights
        self.W_ll_cmb = np.zeros([len(self.evalells),len(np.array(freqs))])
        #rayleigh scattering cross correlation weights
        self.W_ll_rsx = np.zeros([len(self.evalells),len(np.array(freqs))])
        #rayleigh scattering cross correlation weights NO foregrounds
        self.W_ll_rsx_NF = np.zeros([len(self.evalells),len(np.array(freqs))])
        #CMB weights NO foregrounds
        self.W_ll_cmb_NF = np.zeros([len(self.evalells),len(np.array(freqs))])
        #thermal SZ constraining the CIB weights 
        self.W_ll_tsz_c_cib = np.zeros([len(self.evalells),len(np.array(freqs))])
        #thermal SZ constraining the CMB weights 
        self.W_ll_tsz_c_cmb = np.zeros([len(self.evalells),len(np.array(freqs))])
        #CMB constraining the thermal SZ weights 
        self.W_ll_cmb_c_tsz = np.zeros([len(self.evalells),len(np.array(freqs))])
        self.freq = freqs

        #frequency functions for
        f_nu_tsz = f_nu(self.cc.c,np.array(freqs)) #tSZ
        f_nu_cmb = f_nu_tsz*0.0 + 1. #CMB
        f_nu_cib = self.fgs.f_nu_cib(np.array(freqs)) #CIB
        f_nu_rsx = self.fgs.rs_nu(np.array(freqs)) #Rayleigh Cross

        for ii in range(len(self.evalells)):

            cmb_els = fq_mat*0.0 + self.cc.clttfunc(self.evalells[ii])
            
            if v3mode < 0:
                inst_noise = ( old_div(noise_func(self.evalells[ii],np.array(fwhms),np.array(rms_noises),lknee,alpha,dimensionless=False), self.cc.c['TCMBmuK']**2.))
                nells = np.diag(inst_noise)
            elif v3mode<=2:
                nells = N_ell_T_LA_full[:,:,ii]/ self.cc.c['TCMBmuK']**2.

            elif v3mode==3:
                ndiags = []
                for ff in range(len(freqs)):
                    inst_noise = old_div(N_ell_T_LA[ff,ii], self.cc.c['TCMBmuK']**2.)
                    ndiags.append(inst_noise)
                nells = np.diag(np.array(ndiags))

            elif v3mode>=5:
                nells = N_ell_T_LA_full[:,:,ii]/ self.cc.c['TCMBmuK']**2.
                

            totfg = (self.fgs.rad_ps(self.evalells[ii],fq_mat,fq_mat_t) + self.fgs.cib_p(self.evalells[ii],fq_mat,fq_mat_t) +
                      self.fgs.cib_c(self.evalells[ii],fq_mat,fq_mat_t) + self.fgs.tSZ_CIB(self.evalells[ii],fq_mat,fq_mat_t)) \
                      / self.cc.c['TCMBmuK']**2. / ((self.evalells[ii]+1.)*self.evalells[ii]) * 2.* np.pi 

            totfgrs = (self.fgs.rad_ps(self.evalells[ii],fq_mat,fq_mat_t) + self.fgs.cib_p(self.evalells[ii],fq_mat,fq_mat_t) +
                       self.fgs.cib_c(self.evalells[ii],fq_mat,fq_mat_t) + self.fgs.rs_auto(self.evalells[ii],fq_mat,fq_mat_t) + \
                       self.fgs.tSZ_CIB(self.evalells[ii],fq_mat,fq_mat_t)) \
                      / self.cc.c['TCMBmuK']**2. / ((self.evalells[ii]+1.)*self.evalells[ii] ) * 2.* np.pi 

            totfg_cib = (self.fgs.rad_ps(self.evalells[ii],fq_mat,fq_mat_t) + self.fgs.tSZ_CIB(self.evalells[ii],fq_mat,fq_mat_t)) \
                      / self.cc.c['TCMBmuK']**2. / ((self.evalells[ii]+1.)*self.evalells[ii]) * 2.* np.pi


            ksz = fq_mat*0.0 + self.fgs.ksz_temp(self.evalells[ii]) / self.cc.c['TCMBmuK']**2. / ((self.evalells[ii]+1.)*self.evalells[ii]) * 2.* np.pi

            tsz = self.fgs.tSZ(self.evalells[ii],fq_mat,fq_mat_t) / self.cc.c['TCMBmuK']**2. / ((self.evalells[ii]+1.)*self.evalells[ii]) * 2.* np.pi            

            cib = (self.fgs.cib_p(self.evalells[ii],fq_mat,fq_mat_t) + self.fgs.cib_c(self.evalells[ii],fq_mat,fq_mat_t)) \
                     / self.cc.c['TCMBmuK']**2. / ((self.evalells[ii]+1.)*self.evalells[ii]) * 2.* np.pi

            N_ll_for_tsz = nells + totfg + cmb_els + ksz 
            N_ll_for_cmb = nells + totfg + tsz + ksz 
            N_ll_for_rsx = nells + totfg + tsz + ksz #+ cmb_els

            N_ll_for_tsz_c_cmb = nells + totfg + ksz 
            N_ll_for_cmb_c_tsz = N_ll_for_tsz_c_cmb 
            N_ll_for_tsz_c_cib = nells + totfg_cib + cmb_els + ksz

            #N_ll_for_tsz_inv = np.linalg.inv(N_ll_for_tsz)
            #N_ll_for_cmb_inv = np.linalg.inv(N_ll_for_cmb)
            N_ll_for_rsx_inv = np.linalg.inv(N_ll_for_rsx)
            N_ll_for_tsz_c_cmb_inv = np.linalg.inv(N_ll_for_tsz_c_cmb)
            N_ll_for_cmb_c_tsz_inv = N_ll_for_tsz_c_cmb_inv
            N_ll_for_tsz_c_cib_inv = np.linalg.inv(N_ll_for_tsz_c_cib)

            N_ll_noFG = nells
            N_ll_noFG_inv = np.linalg.inv(nells)

            self.W_ll_tsz[ii,:]=weightcalculator(f_nu_tsz,N_ll_for_tsz)
            self.W_ll_rsx[ii,:]=doubleweightcalculator(f_nu_rsx,f_nu_cmb,N_ll_for_rsx_inv)
            self.W_ll_rsx_NF[ii,:]=doubleweightcalculator(f_nu_rsx,f_nu_cmb,N_ll_noFG_inv)
            self.W_ll_cmb[ii,:]=weightcalculator(f_nu_cmb,N_ll_for_cmb)
            self.W_ll_cmb_NF[ii,:]=weightcalculator(f_nu_cmb,N_ll_noFG)
            self.N_ll_tsz[ii] = np.dot(np.transpose(self.W_ll_tsz[ii,:]),np.dot(N_ll_for_tsz,self.W_ll_tsz[ii,:]))
            self.N_ll_cmb[ii] = np.dot(np.transpose(self.W_ll_cmb[ii,:]),np.dot(N_ll_for_cmb,self.W_ll_cmb[ii,:]))
            self.N_ll_rsx[ii] = np.dot(np.transpose(self.W_ll_rsx[ii,:]),np.dot(N_ll_for_rsx,self.W_ll_rsx[ii,:]))
            self.N_ll_rsx_NoFG[ii] = np.dot(np.transpose(self.W_ll_rsx_NF[ii,:]),np.dot(N_ll_noFG,self.W_ll_rsx_NF[ii,:]))
            self.N_ll_cmb_NoFG[ii] = np.dot(np.transpose(self.W_ll_cmb_NF[ii,:]),np.dot(N_ll_noFG,self.W_ll_cmb_NF[ii,:]))
            self.W_ll_tsz_c_cmb[ii,:]=constweightcalculator(f_nu_cmb,f_nu_tsz,N_ll_for_tsz_c_cmb_inv)
            self.W_ll_tsz_c_cib[ii,:]=constweightcalculator(f_nu_cib,f_nu_tsz,N_ll_for_tsz_c_cib_inv)
            self.W_ll_cmb_c_tsz[ii,:]=constweightcalculator(f_nu_tsz,f_nu_cmb,N_ll_for_cmb_c_tsz_inv)
            self.N_ll_tsz_c_cmb[ii] = np.dot(np.transpose(self.W_ll_tsz_c_cmb[ii,:]),np.dot(N_ll_for_tsz_c_cmb,self.W_ll_tsz_c_cmb[ii,:]))
            self.N_ll_cmb_c_tsz[ii] = np.dot(np.transpose(self.W_ll_cmb_c_tsz[ii,:]),np.dot(N_ll_for_cmb_c_tsz,self.W_ll_cmb_c_tsz[ii,:]))
            self.N_ll_tsz_c_cib[ii] = np.dot(np.transpose(self.W_ll_tsz_c_cib[ii,:]),np.dot(N_ll_for_tsz_c_cib,self.W_ll_tsz_c_cib[ii,:]))

    def Noise_ellyy(self,constraint='None'):
        if (constraint=='None'):
            return self.evalells,self.N_ll_tsz
        elif (constraint=='cmb'):
            return self.evalells,self.N_ll_tsz_c_cmb
        elif (constraint=='cib'):
            return self.evalells,self.N_ll_tsz_c_cib
        else:
            return "Wrong option"

    def Noise_ellcmb(self,constraint='None'):
        if (constraint=='None'):
            return self.evalells,self.N_ll_cmb
        elif (constraint=='tsz'):
            return self.evalells,self.N_ll_cmb_c_tsz
        else:
            return "Wrong option"

    def Noise_ellrsx(self,option='None'):
        if (option=='None'):
            return self.evalells,self.N_ll_rsx
        elif (option=='NoILC'):
            return self.evalells,self.N_ll_rsx_NoFG
        else:
            return "Wrong option"

    def Forecast_Cellyy(self,ellBinEdges,fsky,constraint='None'):

        ellMids  =  (ellBinEdges[1:] + ellBinEdges[:-1])/ 2

        cls_tsz = self.fgs.tSZ(self.evalells,self.freq[0],self.freq[0]) / self.cc.c['TCMBmuK']**2. \
                / ((self.evalells+1.)*self.evalells) * 2.* np.pi

        cls_yy = cls_tsz /(f_nu(self.cc.c,self.freq[0]))**2  # Normalized to get Cell^yy

        LF = LensForecast()
        if (constraint=='None'):
            LF.loadGenericCls("yy",self.evalells,cls_yy,self.evalells,self.N_ll_tsz)
        elif (constraint=='cmb'):
            LF.loadGenericCls("yy",self.evalells,cls_yy,self.evalells,self.N_ll_tsz_c_cmb)
        elif (constraint=='cib'):
            LF.loadGenericCls("yy",self.evalells,cls_yy,self.evalells,self.N_ll_tsz_c_cib)
        else:
            return "Wrong option"

        sn,errs = LF.sn(ellBinEdges,fsky,"yy") # not squared

        cls_out = np.interp(ellMids,self.evalells,cls_yy)

        return ellMids,cls_out,errs,sn

    def Forecast_Cellcmb(self,ellBinEdges,fsky,constraint='None'):

        ellMids  =  (ellBinEdges[1:] + ellBinEdges[:-1])/ 2.

        cls_cmb = self.cc.clttfunc(self.evalells)

        LF = LensForecast()
        if (constraint=='None'):
            LF.loadGenericCls("tt",self.evalells,cls_cmb,self.evalells,self.N_ll_cmb)
        elif (constraint=='tsz'):
            LF.loadGenericCls("tt",self.evalells,cls_cmb,self.evalells,self.N_ll_cmb_c_tsz)
        else:
            return "Wrong option"

        sn,errs = LF.sn(ellBinEdges,fsky,"tt") # not squared

        cls_out = np.interp(ellMids,self.evalells,cls_cmb)

        return ellMids,cls_out,errs,sn

    def Forecast_Cellrsx(self,ellBinEdges,fsky,option='None'):

        ellMids  =  (ellBinEdges[1:] + ellBinEdges[:-1])/ 2.
        ellWidth =  (ellBinEdges[1:] - ellBinEdges[:-1])

        #cls_rsx = self.fgs.rs_cross(self.evalells,self.freq[0]) / self.cc.c['TCMBmuK']**2.\
        #        / ((self.evalells+1.)*self.evalells) * 2.* np.pi

        #print (cls_rsx, self.freq[0],self.fgs.nu_rs)
        
        cls_rsx = self.fgs.rs_cross(self.evalells,self.fgs.nu_rs) / self.cc.c['TCMBmuK']**2.\
                / ((self.evalells+1.)*self.evalells) * 2.* np.pi

        #cls_rs = self.fgs.rs_auto(self.evalells,self.fgs.nu_rs,self.fgs.nu_rs) / self.cc.c['TCMBmuK']**2.\
        #        / ((self.evalells+1.)*self.evalells) * 2.* np.pi
        
        #cls_cmb = self.cc.clttfunc(self.evalells)

        LF = LensForecast()

        if (option=='None'):        
            LF.loadGenericCls("rr",self.evalells,cls_rsx,self.evalells,self.N_ll_rsx)
            #LF.loadGenericCls("tt",self.evalells,cls_cmb,self.evalells,self.N_ll_cmb)
            Nellrsx = self.N_ll_rsx
            #Nellcmb = self.N_ll_cmb
        elif (option=='NoILC'):
            LF.loadGenericCls("rr",self.evalells,cls_rsx,self.evalells,self.N_ll_rsx_NoFG)
            #LF.loadGenericCls("tt",self.evalells,cls_cmb,self.evalells,self.N_ll_rsx_NoFG)
            Nellrsx = self.N_ll_rsx_NoFG
            #Nellcmb = self.N_ll_cmb_NoFG
        else:
            return "Wrong option"


        #vars, _, _ = LF.KnoxCov("rr","tt",ellBinEdges,fsky)

        sn,errs = LF.sn(ellBinEdges,fsky,"rr") # not squared 

        #sn = np.sqrt ( fsky / 2 * np.sum( (2.*self.evalells + 1) * (cls_rsx**2) / ((cls_rs + Nellrsx)* (cls_cmb + Nellcmb)) ) ) 
        #errs = 1.#np.sqrt(vars)

        cls_out = np.interp(ellMids,self.evalells,cls_rsx)

        #errs = cls_out * 0.0 + 1.

        return ellMids,cls_out,errs,sn


    def PlotyWeights(self,outfile):
        
        #plot weights
        pl = Plotter()
        for ii in range(len(self.freq)):
            pl.add(self.evalells,self.W_ll_tsz[:,ii],label=str(self.freq[ii])+' GHz')
        pl.legend(loc='lower left',labsize=10)
        pl.done(outfile)

    def PlotcmbWeights(self,outfile):
        
        #plot weights
        pl = Plotter()
        for ii in range(len(self.freq)):
            pl.add(self.evalells,self.W_ll_cmb[:,ii],label=str(self.freq[ii])+' GHz')
        pl.legend(loc='lower left',labsize=10)
        pl.done(outfile)

class Filters(object):
    def __init__(self):
        self.disc_fac = np.sqrt(2)

    def inner_app (self,ell,theta_a):
        theta_a /= 60. #arcmin to degs
        theta_a *= old_div(np.pi,180.) #degs to rad 
        xx = ell*theta_a
        ans = 2./(xx)*j1(xx)
        return ans

    def outer_app (self,ell,theta_a,theta_b):
        theta_a /= 60. #arcmin to degs
        theta_a *= old_div(np.pi,180.) #degs to rad 
        theta_b /= 60. #arcmin to degs
        theta_b *= old_div(np.pi,180.) #degs to rad 
        xx = ell*theta_a
        yy = ell*theta_b
        ans = 2./(yy*theta_b - xx*theta_a)*(theta_b*j1(yy)- theta_a*j1(xx))
        return ans

    def AP_filter (self,ell,theta_a,theta_b):
        ans = self.inner_app(ell,theta_a)-self.outer_app(ell,theta_a,theta_b)
        return ans

    def filter_var (self,theta1,theta2,ells,Nell,beam=1.):
        if (beam != 1):
            beam = self.beam_func(ells,beam)

        if (theta1 == theta2):
            ans  = np.trapz(ells*Nell*beam**2*self.AP_filter(ells,theta1,self.disc_fac*theta1)**2,dx=np.diff(ells))
            ans  /= 2*np.pi
        else:
            var1  = np.trapz(ells*Nell*beam**2*self.AP_filter(ells,theta1,self.disc_fac*theta1)**2,dx=np.diff(ells))
            var2  = np.trapz(ells*Nell*beam**2*self.AP_filter(ells,theta2,self.disc_fac*theta2)**2,dx=np.diff(ells))
            var12 = np.trapz(ells*Nell*beam**2*self.AP_filter(ells,theta1,self.disc_fac*theta1)
                       *self.AP_filter(ells,theta2,self.disc_fac*theta2),dx=np.diff(ells))
            #ans = old_div((var1 + var2 - 2.*var12),(2.*np.pi))
            ans = (var1 + var2 - 2.*var12)/(2.*np.pi)
        return ans

    def beam_func(self,ell,theta_b):
        theta_b /= 60.
        theta_b *= np.pi/180.
        ans = np.exp(-1.0*ell**2*theta_b**2/(16.*np.log(2.0)))
        return ans

#    def variance(self,ell,theta,disc_fac,cltot):
#        cl_var = np.sqrt(np.sum(self.beam_func(ell,theta)**2 * cltot * self.inner_app(ell,theta)**2) \
#                             +np.sum(self.beam_func(ell,theta)**2 * cltot * self.outer_app(ell,theta,theta*disc_fac)**2) \
#                             -2.0*np.sum(self.beam_func(ell,theta)**2 * cltot * self.inner_app(ell,theta) \
#                                             * self.outer_app(ell,theta,theta*disc_fac)))
#        return cl_var

class ILC_simple_pol(object):
    def __init__(self,clusterCosmology, \
                 fwhms=[1.5],rms_noises =[1.], freqs = [150.],lmax=8000,lknee=0.,alpha=1., \
                 dell=1.,ksz_file='input/ksz_BBPS.txt',ksz_p_file='input/ksz_p_BBPS.txt', \
                 tsz_cib_file='input/sz_x_cib_template.dat',fg=True):

        self.cc = clusterCosmology

        if (len(freqs) > 1):
            fq_mat   = np.matlib.repmat(freqs,len(freqs),1) 
            fq_mat_t = np.transpose(np.matlib.repmat(freqs,len(freqs),1))
        else:
            fq_mat   = freqs
            fq_mat_t = freqs

        self.fgs = fgNoises(self.cc.c,ksz_file=ksz_file,ksz_p_file=ksz_p_file,tsz_cib_file=tsz_cib_file,tsz_battaglia_template_csv="data/sz_template_battaglia.csv")
        
        self.dell = dell
        self.evalells = np.arange(2,lmax,self.dell)
        self.N_ll_cmb = self.evalells*0.0

        self.W_ll_cmb = np.zeros([len(self.evalells),len(np.array(freqs))])
        self.freq = freqs

        f_nu_cmb = f_nu_tsz*0.0 + 1.

        for ii in range(len(self.evalells)):
            cmb_els = fq_mat*0.0 + self.cc.cleefunc(self.evalells[ii])
            ## MAKE POL NOISE
            inst_noise = ( old_div(noise_func(self.evalells[ii],np.array(fwhms),np.array(rms_noises),lknee,alpha,dimensionless=False), self.cc.c['TCMBmuK']**2.))
        
            nells = np.diag(inst_noise)
            
            totfg = (self.fgs.rad_pol_ps(self.evalells[ii],fq_mat,fq_mat_t) + \
                         self.fgs.gal_dust_pol(self.evalells[ii],fq_mat,fq_mat_t) + \
                         self.fgs.gal_sync_pol(self.evalells[ii],fq_mat,fq_mat_t))

            N_ll_for_cmb = nells + totfg

            N_ll_for_cmb_inv = np.linalg.inv(N_ll_for_cmb)

            self.W_ll_cmb[ii,:] = 1./np.dot(np.transpose(f_nu_cmb),np.dot(N_ll_for_cmb_inv,f_nu_cmb)) \
                                  * np.dot(np.transpose(f_nu_cmb),N_ll_for_cmb_inv)

            self.N_ll_cmb[ii] = np.dot(np.transpose(self.W_ll_cmb[ii,:]),np.dot(N_ll_for_cmb,self.W_ll_cmb[ii,:]))





