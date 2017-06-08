import numpy as np
from sympy.functions import coth
from scipy.interpolate import interp1d
from orphics.tools.cmb import noise_func
from szar.foregrounds import fgNoises
from szar.counts import f_nu
import numpy.matlib

class ILC_simple:
    def __init__(self,clusterCosmology, \
                 fwhms=[1.5],rms_noises =[1.], freqs = [150.],lmax=8000,lknee=0.,alpha=1., \
                 dell=1000.,ksz_file='input/ksz_BBPS.txt',ksz_p_file='input/ksz_p_BBPS.txt', \
                 tsz_cib_file='input/sz_x_cib_template.dat' fg=True):

        self.cc = clusterCosmology

        if (len(freqs) > 1):
            fq_mat   = np.matlib.repmat(freqs,len(freqs),1) 
            fq_mat_t = np.transpose(np.matlib.repmat(freqs,len(freqs),1))
        else:
            fq_mat   = freqs
            fq_mat_t = freqs

        fgs = fgNoises(self.cc.c,ksz_file=ksz_file,ksz_p_file=ksz_p_file,tsz_cib_file=tsz_cib_file)

        self.dell = dell
        self.evalells = np.arange(2,lmax,self.dell)
        self.N_ll_tsz = self.evalells*0.0
        self.N_ll_cmb = self.evalells*0.0
        self.W_ll_tsz = np.array(len(self.evalells),len(freqs))
        self.W_ll_cmb = np.array(len(self.evalells),len(freqs))
        self.f_nu_arr = np.array(freqs)*0.0

        b_0 = 1.

        for ii in xrange(len(freqs)):
            self.f_nu_arr[ii] = f_nu(self.cc.c,freqs[ii])

        for ii in xrange(len(self.evalells)):

            cmb_els = fq_mat*0.0 + self.cc.clttfunc(self.evalells[ii])
            
            inst_noise = ( noise_func(self.evalells[ii],fwhm,noise,lknee,alpha) / self.cc.c['TCMBmuK']**2.)
        
            nells = inst_noise

            totfg = ((fgs.rad_ps(self.evalells[ii],fq_mat,fq_mat_t) + fgs.cib_p(self.evalells[ii],fq_mat,fq_mat_t) +
                      fgs.cib_c(self.evalells[ii],fq_mat,fq_mat_t) + fgs.tSZ_CIB(self.evalells[ii],fq_mat,fq_mat_t))
                      / self.cc.c['TCMBmuK']**2. / ((self.evalells[ii]+1.)*self.evalells[ii]) * 2.* np.pi )

            ksz = fq_mat*0.0 + fgs.ksz_temp(self.evalells[ii])) / self.cc.c['TCMBmuK']**2. / ((self.evalells[ii]+1.)*self.evalells[ii]) * 2.* np.pi

            tsz = fgs.tSZ(self.evalells[ii],fq_mat,fq_mat_t) / self.cc.c['TCMBmuK']**2. / ((self.evalells[ii]+1.)*self.evalells[ii]) * 2.* np.pi            

            N_ll_for_tsz = totfg + cmb_els + ksz 
            N_ll_for_cmb = totfg + tsz
        
            f_nu_arr = f_nu(self.cc.c,np.array(freqs))
            f_nu_bm = f_nu_arr

            N_ll_for_tsz_inv = np.linalg.inv(N_ll_for_tsz)
            N_ll_for_cmb_inv = np.linalg.inv(N_ll_for_cmb)

            self.W_ll_tsz[ii,:] = b_0 * 1./np.dot(np.transpose(f_nu_bm),np.dot(N_ll_for_tsz_inv,f_nu_bm)) \
                                  * np.dot(np.transpose(f_nu_bm),N_ll_for_tsz_inv)
            self.W_ll_cmb[ii,:] = b_0 * 1./np.dot(np.transpose(f_nu_bm),np.dot(N_ll_for_cmb_inv,f_nu_bm)) \
                                  * np.dot(np.transpose(f_nu_bm),N_ll_for_cmb_inv)

            self.N_ll_tsz[ii] = np.dot(np.transpose(self.W_ll_tsz[ii,:]),np.dot(N_ll_for_tsz,self.W_ll_tsz[ii,:]))
            self.N_ll_cmb[ii] = np.dot(np.transpose(self.W_ll_cmb[ii,:]),np.dot(N_ll_for_cmb,self.W_ll_cmb[ii,:]))
