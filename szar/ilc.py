import numpy as np
from sympy.functions import coth
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
        self.W_ll_tsz = np.zeros([len(self.evalells),len(np.array(freqs))])
        self.W_ll_cmb = np.zeros([len(self.evalells),len(np.array(freqs))])
        self.freq = freqs

        f_nu_tsz = f_nu(self.cc.c,np.array(freqs)) #* self.cc.c['TCMBmuK']
        f_nu_cmb = f_nu_tsz*0.0 + 1.

        for ii in xrange(len(self.evalells)):

            cmb_els = fq_mat*0.0 + self.cc.clttfunc(self.evalells[ii])
            

            inst_noise = ( noise_func(self.evalells[ii],np.array(fwhms),np.array(rms_noises),lknee,alpha) / self.cc.c['TCMBmuK']**2.)
        
            nells = np.diag(inst_noise)#np.outer(inst_noise,inst_noise)
            #nells = np.outer(inst_noise,inst_noise)
            

            totfg = (self.fgs.rad_ps(self.evalells[ii],fq_mat,fq_mat_t) + self.fgs.cib_p(self.evalells[ii],fq_mat,fq_mat_t) +
                      self.fgs.cib_c(self.evalells[ii],fq_mat,fq_mat_t) + self.fgs.tSZ_CIB(self.evalells[ii],fq_mat,fq_mat_t)) \
                      / self.cc.c['TCMBmuK']**2. / ((self.evalells[ii]+1.)*self.evalells[ii]) * 2.* np.pi 

            ksz = fq_mat*0.0 + self.fgs.ksz_temp(self.evalells[ii]) / self.cc.c['TCMBmuK']**2. / ((self.evalells[ii]+1.)*self.evalells[ii]) * 2.* np.pi

            tsz = self.fgs.tSZ(self.evalells[ii],fq_mat,fq_mat_t) / self.cc.c['TCMBmuK']**2. / ((self.evalells[ii]+1.)*self.evalells[ii]) * 2.* np.pi            
            N_ll_for_tsz = nells + totfg + cmb_els + ksz 
            N_ll_for_cmb = nells + totfg + tsz
        
            N_ll_for_tsz_inv = np.linalg.inv(N_ll_for_tsz)
            N_ll_for_cmb_inv = np.linalg.inv(N_ll_for_cmb)

            self.W_ll_tsz[ii,:] = 1./np.dot(np.transpose(f_nu_tsz),np.dot(N_ll_for_tsz_inv,f_nu_tsz)) \
                                  * np.dot(np.transpose(f_nu_tsz),N_ll_for_tsz_inv)
            self.W_ll_cmb[ii,:] = 1./np.dot(np.transpose(f_nu_cmb),np.dot(N_ll_for_cmb_inv,f_nu_cmb)) \
                                  * np.dot(np.transpose(f_nu_cmb),N_ll_for_cmb_inv)

            self.N_ll_tsz[ii] = np.dot(np.transpose(self.W_ll_tsz[ii,:]),np.dot(N_ll_for_tsz,self.W_ll_tsz[ii,:])) #* self.cc.c['TCMBmuK']**2.
            self.N_ll_cmb[ii] = np.dot(np.transpose(self.W_ll_cmb[ii,:]),np.dot(N_ll_for_cmb,self.W_ll_cmb[ii,:]))

            if (ii == 3000):
                print "NOISE"
                print 'ell', self.evalells[ii]
                print 'inst', nells[4,4] * self.evalells[ii]**2
                print 'cmb',  cmb_els[4,4] * self.evalells[ii]**2
                print 'fg', totfg[4,4] * self.evalells[ii]**2
                print 'ksz', ksz[4,4] * self.evalells[ii]**2
                print 'tsz', tsz[4,4] * self.evalells[ii]**2
                print 'noise mat', N_ll_for_tsz[4,4] * self.evalells[ii]**2
                print 'wieghts', self.W_ll_tsz[ii,:]
                print 'reduced noise', self.N_ll_tsz[ii] * self.evalells[ii]**2 #* self.cc.c['TCMBmuK']**2.
                

    def Forecast_Cellyy(self,ellBinEdges,fsky):

        ellMids  =  (ellBinEdges[1:] + ellBinEdges[:-1]) / 2

        cls_tsz = self.fgs.tSZ(self.evalells,self.freq[0],self.freq[0]) / self.cc.c['TCMBmuK']**2. \
                / ((self.evalells+1.)*self.evalells) * 2.* np.pi

        cls_yy = cls_tsz / (f_nu(self.cc.c,self.freq[0]))**2  # Normalized to get Cell^yy

#        LF = orphics.theory.gaussianCov.LensForecast()
        LF = LensForecast()
        LF.loadGenericCls("yy",self.evalells,cls_yy,self.evalells,self.N_ll_tsz)

        sn = LF.sn(ellBinEdges,fsky,"yy")
        errs2 = LF.sigmaClSquared("yy",ellBinEdges,fsky)

        cls_out = np.interp(ellMids,self.evalells,cls_yy)

        return ellMids,cls_out,np.sqrt(errs2),sn

    def Forecast_Cellcmb(self,ellBinEdges,fsky):

        ellMids  =  (ellBinEdges[1:] + ellBinEdges[:-1]) / 2

        cls_cmb = self.cc.clttfunc(self.evalells)

#        LF = orphics.thoery.gaussianCov.LensForecast()
        LF = LensForecast()
        LF.loadGenericCls("tt",self.evalells,cls_cmb,self.evalells,self.N_ll_cmb)

        sn = LF.sn(ellBinEdges,fsky,"tt")
        errs2 = LF.sigmaClSquared("tt",ellBinEdges,fsky)

        cls_out = np.interp(ellMids,self.evalells,cls_cmb)

        return ellMids,cls_out,np.sqrt(errs2),sn

    def PlotyWeights(self):
        
        self.W_ll_cmb[:,]

        #plot
        pl = Plotter()
        pl._ax.set_ylim(-3.6,-0.8)
        pl.add(np.log10(M),np.log10(N[:,0]*M/(self.cc.rhoc0om)),color='black')
        pl.add(np.log10(M),np.log10(N_8[:,0]*M/(self.cc.rhoc0om)),color='black')
        pl.add(np.log10(M),np.log10(N_32[:,0]*M/(self.cc.rhoc0om)),color='black')
        pl.done("output/tinkervals.png")
