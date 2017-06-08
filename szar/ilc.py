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
                 dell=1,ksz_file='input/ksz_BBPS.txt',ksz_p_file='input/ksz_p_BBPS.txt', \
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

        cmb_els = self.cc.clttfunc(self.evalells)
        inst_noise = ( noise_func(self.evalells,fwhm,noise,lknee,alpha) / self.cc.c['TCMBmuK']**2.)
        
        nells = inst_noise

        totfg = (fgs.rad_ps(self.evalells,freq,freq) + fgs.cib_p(self.evalells,freq,freq) + \
                   fgs.cib_c(self.evalells,freq,freq) + fgs.tSZ_CIB(self.evalells,freq,freq)) \
                   / self.cc.c['TCMBmuK']**2. / ((self.evalells+1.)*self.evalells) * 2.* np.pi

        ksz   = fgs.ksz_temp(self.evalells)) / cl_factor

        
