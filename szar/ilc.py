import numpy as np
from scipy.interpolate import interp1d
from orphics.tools.cmb import noise_func
from orphics.theory.gaussianCov import LensForecast
from szar.foregrounds import fgNoises, f_nu
from orphics.tools.io import Plotter
import numpy.matlib

class ILC_simple:
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
        self.N_ll_tsz = self.evalells*0.0
        self.N_ll_cmb = self.evalells*0.0
        self.N_ll_tsz = self.evalells*0.0
        self.N_ll_cmb_c_tsz = self.evalells*0.0
        self.N_ll_tsz_c_cmb = self.evalells*0.0
        self.N_ll_tsz_c_cib = self.evalells*0.0

        self.W_ll_tsz = np.zeros([len(self.evalells),len(np.array(freqs))])
        self.W_ll_cmb = np.zeros([len(self.evalells),len(np.array(freqs))])
        self.W_ll_tsz_c_cib = np.zeros([len(self.evalells),len(np.array(freqs))])
        self.W_ll_tsz_c_cmb = np.zeros([len(self.evalells),len(np.array(freqs))])
        self.W_ll_cmb_c_tsz = np.zeros([len(self.evalells),len(np.array(freqs))])
        self.freq = freqs

        f_nu_tsz = f_nu(self.cc.c,np.array(freqs)) 
        f_nu_cmb = f_nu_tsz*0.0 + 1.
        f_nu_cib = self.fgs.f_nu_cib(np.array(freqs))

        for ii in range(len(self.evalells)):

            cmb_els = fq_mat*0.0 + self.cc.clttfunc(self.evalells[ii])

            inst_noise = ( noise_func(self.evalells[ii],np.array(fwhms),np.array(rms_noises),lknee,alpha,dimensionless=False) / self.cc.c['TCMBmuK']**2.)
        
            nells = np.diag(inst_noise)
            
            totfg = (self.fgs.rad_ps(self.evalells[ii],fq_mat,fq_mat_t) + self.fgs.cib_p(self.evalells[ii],fq_mat,fq_mat_t) +
                      self.fgs.cib_c(self.evalells[ii],fq_mat,fq_mat_t) + self.fgs.tSZ_CIB(self.evalells[ii],fq_mat,fq_mat_t)) \
                      / self.cc.c['TCMBmuK']**2. / ((self.evalells[ii]+1.)*self.evalells[ii]) * 2.* np.pi 

            totfg_cib = (self.fgs.rad_ps(self.evalells[ii],fq_mat,fq_mat_t) + self.fgs.tSZ_CIB(self.evalells[ii],fq_mat,fq_mat_t)) \
                      / self.cc.c['TCMBmuK']**2. / ((self.evalells[ii]+1.)*self.evalells[ii]) * 2.* np.pi


            ksz = fq_mat*0.0 + self.fgs.ksz_temp(self.evalells[ii]) / self.cc.c['TCMBmuK']**2. / ((self.evalells[ii]+1.)*self.evalells[ii]) * 2.* np.pi

            tsz = self.fgs.tSZ(self.evalells[ii],fq_mat,fq_mat_t) / self.cc.c['TCMBmuK']**2. / ((self.evalells[ii]+1.)*self.evalells[ii]) * 2.* np.pi            

            cib = (self.fgs.cib_p(self.evalells[ii],fq_mat,fq_mat_t) + self.fgs.cib_c(self.evalells[ii],fq_mat,fq_mat_t)) \
                     / self.cc.c['TCMBmuK']**2. / ((self.evalells[ii]+1.)*self.evalells[ii]) * 2.* np.pi

            N_ll_for_tsz = nells + totfg + cmb_els + ksz 
            N_ll_for_cmb = nells + totfg + tsz + ksz

            N_ll_for_tsz_c_cmb = nells + totfg 
            N_ll_for_cmb_c_tsz = N_ll_for_tsz_c_cmb + ksz
            N_ll_for_tsz_c_cib = nells + totfg_cib + cmb_els + ksz

            N_ll_for_tsz_inv = np.linalg.inv(N_ll_for_tsz)
            N_ll_for_cmb_inv = np.linalg.inv(N_ll_for_cmb)
            N_ll_for_tsz_c_cmb_inv = np.linalg.inv(N_ll_for_tsz_c_cmb)
            N_ll_for_cmb_c_tsz_inv = N_ll_for_tsz_c_cmb_inv
            N_ll_for_tsz_c_cib_inv = np.linalg.inv(N_ll_for_tsz_c_cib)


            self.W_ll_tsz[ii,:] = 1./np.dot(np.transpose(f_nu_tsz),np.dot(N_ll_for_tsz_inv,f_nu_tsz)) \
                                  * np.dot(np.transpose(f_nu_tsz),N_ll_for_tsz_inv)
            self.W_ll_cmb[ii,:] = 1./np.dot(np.transpose(f_nu_cmb),np.dot(N_ll_for_cmb_inv,f_nu_cmb)) \
                                  * np.dot(np.transpose(f_nu_cmb),N_ll_for_cmb_inv)

            self.N_ll_tsz[ii] = np.dot(np.transpose(self.W_ll_tsz[ii,:]),np.dot(N_ll_for_tsz,self.W_ll_tsz[ii,:]))
            self.N_ll_cmb[ii] = np.dot(np.transpose(self.W_ll_cmb[ii,:]),np.dot(N_ll_for_cmb,self.W_ll_cmb[ii,:]))

            self.W_ll_tsz_c_cmb[ii,:] = (np.dot(np.transpose(f_nu_cmb),np.dot(N_ll_for_tsz_c_cmb_inv,f_nu_cmb)) \
                                             * np.dot(np.transpose(f_nu_tsz),N_ll_for_tsz_c_cmb_inv) \
                                             - np.dot(np.transpose(f_nu_tsz),np.dot(N_ll_for_tsz_c_cmb_inv,f_nu_cmb)) \
                                             * np.dot(np.transpose(f_nu_cmb),N_ll_for_tsz_c_cmb_inv)) / \
                                        (np.dot(np.transpose(f_nu_cmb),np.dot(N_ll_for_tsz_c_cmb_inv,f_nu_cmb)) \
                                             * np.dot(np.transpose(f_nu_tsz),np.dot(N_ll_for_tsz_c_cmb_inv,f_nu_tsz)) \
                                             - (np.dot(np.transpose(f_nu_tsz),np.dot(N_ll_for_tsz_c_cmb_inv,f_nu_cmb)))**2)

            self.W_ll_tsz_c_cib[ii,:] = (np.dot(np.transpose(f_nu_cib),np.dot(N_ll_for_tsz_c_cib_inv,f_nu_cib)) \
                                             * np.dot(np.transpose(f_nu_tsz),N_ll_for_tsz_c_cib_inv) \
                                             - np.dot(np.transpose(f_nu_tsz),np.dot(N_ll_for_tsz_c_cib_inv,f_nu_cib)) \
                                             * np.dot(np.transpose(f_nu_cib),N_ll_for_tsz_c_cib_inv)) / \
                                        (np.dot(np.transpose(f_nu_cib),np.dot(N_ll_for_tsz_c_cib_inv,f_nu_cib)) \
                                             * np.dot(np.transpose(f_nu_tsz),np.dot(N_ll_for_tsz_c_cib_inv,f_nu_tsz)) \
                                             - (np.dot(np.transpose(f_nu_tsz),np.dot(N_ll_for_tsz_c_cib_inv,f_nu_cib)))**2)

            self.W_ll_cmb_c_tsz[ii,:] = (np.dot(np.transpose(f_nu_tsz),np.dot(N_ll_for_cmb_c_tsz_inv,f_nu_tsz)) \
                                             * np.dot(np.transpose(f_nu_cmb),N_ll_for_cmb_c_tsz_inv) \
                                             - np.dot(np.transpose(f_nu_cmb),np.dot(N_ll_for_cmb_c_tsz_inv,f_nu_tsz)) \
                                             * np.dot(np.transpose(f_nu_tsz),N_ll_for_cmb_c_tsz_inv)) / \
                                        (np.dot(np.transpose(f_nu_cmb),np.dot(N_ll_for_cmb_c_tsz_inv,f_nu_cmb)) \
                                             * np.dot(np.transpose(f_nu_tsz),np.dot(N_ll_for_cmb_c_tsz_inv,f_nu_tsz)) \
                                             - (np.dot(np.transpose(f_nu_cmb),np.dot(N_ll_for_cmb_c_tsz_inv,f_nu_tsz)))**2)

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

    def Forecast_Cellyy(self,ellBinEdges,fsky,constraint='None'):

        ellMids  =  (ellBinEdges[1:] + ellBinEdges[:-1]) / 2

        cls_tsz = self.fgs.tSZ(self.evalells,self.freq[0],self.freq[0]) / self.cc.c['TCMBmuK']**2. \
                / ((self.evalells+1.)*self.evalells) * 2.* np.pi

        cls_yy = cls_tsz / (f_nu(self.cc.c,self.freq[0]))**2  # Normalized to get Cell^yy

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

        ellMids  =  (ellBinEdges[1:] + ellBinEdges[:-1]) / 2

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

    def PlotyWeights(self,outfile):
        
        #plot weights
        pl = Plotter()
        for ii in range(len(self.freq)):
            pl.add(self.evalells,self.W_ll_tsz[:,ii],label=str(self.freq[ii])+' GHz')
        pl.legendOn(loc='lower left',labsize=10)
        pl.done(outfile)

    def PlotcmbWeights(self,outfile):
        
        #plot weights
        pl = Plotter()
        for ii in range(len(self.freq)):
            pl.add(self.evalells,self.W_ll_cmb[:,ii],label=str(self.freq[ii])+' GHz')
        pl.legendOn(loc='lower left',labsize=10)
        pl.done(outfile)

    def inner_app (self,ell,theta_a):
        theta_a /= 60. #arcmin to degs
        theta_a *= np.pi/180. #degs to rad 
        xx = ell*theta_a
        ans = 2./(xx)*j1(xx)
        return ans

    def outer_app (self,ell,theta_a,theta_b):
        theta_a /= 60. #arcmin to degs
        theta_a *= np.pi/180. #degs to rad 
        theta_b /= 60. #arcmin to degs
        theta_b *= np.pi/180. #degs to rad 
        xx = ell*theta_a
        yy = ell*theta_b
        ans = 2./(yy*theta_b - xx*theta_a)*(theta_b*j1(yy)- theta_a*j1(xx))
        return ans
