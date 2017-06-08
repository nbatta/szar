import matplotlib
matplotlib.use('Agg')
import numpy as np
from szar.counts import ClusterCosmology,f_nu
from szar.szproperties import SZ_Cluster_Model
from szar.foregrounds import fgNoises
import sys,os
from ConfigParser import SafeConfigParser 
import cPickle as pickle
from orphics.tools.io import dictFromSection, listFromConfig, Plotter
from orphics.tools.cmb import noise_func

iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)

constDict = dictFromSection(Config,'constants')
clusterDict = dictFromSection(Config,'cluster_params')
fparams = {}   # the 
for (key, val) in Config.items('params'):
    if ',' in val:
        param, step = val.split(',')
        fparams[key] = float(param)
    else:
        fparams[key] = float(val)

clttfile = Config.get('general','clttfile')
cc = ClusterCosmology(fparams,constDict,lmax=8000,pickling=True)#clTTFixFile=clttfile)

fgs = fgNoises(cc.c,ksz_battaglia_test_csv="data/ksz_template_battaglia.csv",tsz_battaglia_template_csv="data/sz_template_battaglia.csv")


experimentName = "SO-6m"
beams = listFromConfig(Config,experimentName,'beams')
noises = listFromConfig(Config,experimentName,'noises')
freqs = listFromConfig(Config,experimentName,'freqs')
lmax = int(Config.getfloat(experimentName,'lmax'))
lknee = listFromConfig(Config,experimentName,'lknee')[0]
alpha = listFromConfig(Config,experimentName,'alpha')[0]
fsky = Config.getfloat(experimentName,'fsky')

SZProfExample = SZ_Cluster_Model(clusterCosmology=cc,clusterDict=clusterDict,rms_noises = noises,fwhms=beams,freqs=freqs,lmax=lmax,lknee=lknee,alpha=alpha)


#outDir = os.environ['WWW']+"plots/"
outDir = "tests/"
ls = np.arange(2,8000,10)

ksz = fgs.ksz_temp(ls)/ls/(ls+1.)*2.*np.pi/ cc.c['TCMBmuK']**2.
# kszAlt = fgs.ksz_battaglia_test(ls)/ls/(ls+1.)*2.*np.pi/ cc.c['TCMBmuK']**2.

print_ells = [100,200,300,400,500,600]

fq_mat   = np.matlib.repmat(freqs,len(freqs),1)
fq_mat_t = np.transpose(np.matlib.repmat(freqs,len(freqs),1))

f_nu_arr = np.array(freqs)*0.0
for ii in xrange(len(freqs)):
    f_nu_arr[ii] = f_nu(cc.c,freqs[ii])

#print fq_mat
#print fq_mat_t

radio_mat = fgs.rad_ps(print_ells[4],fq_mat,fq_mat_t) / cc.c['TCMBmuK']**2.

print "contraction", np.dot(np.transpose(f_nu_arr),np.dot(np.linalg.inv(radio_mat),f_nu_arr))

#print fgs.rad_ps(ls[10],fq_mat_t,fq_mat)/ls[10]/(ls[10]+1.)*2.*np.pi/ cc.c['TCMBmuK']**2.

#print fgs.rad_ps(ls[10],fq_mat_t,fq_mat)*0.0 + 1.

print "noise", noise_func(print_ells[4],np.array(beams),np.array(noises),lknee,alpha) / cc.c['TCMBmuK']**2.

for fwhm,noiseT,testFreq in zip(beams,noises,freqs):
    totCl = 0.
    #print testFreq
    noise = noise_func(ls,fwhm,noiseT,lknee,alpha) / cc.c['TCMBmuK']**2.
    
    radio = fgs.rad_ps(ls,testFreq,testFreq)/ls/(ls+1.)*2.*np.pi/ cc.c['TCMBmuK']**2.
    cibp = fgs.cib_p(ls,testFreq,testFreq) /ls/(ls+1.)*2.*np.pi/ cc.c['TCMBmuK']**2.
    cibc = fgs.cib_c(ls,testFreq,testFreq)/ls/(ls+1.)*2.*np.pi/ cc.c['TCMBmuK']**2.
    tsz = fgs.tSZ(ls,testFreq,testFreq)/ls/(ls+1.)*2.*np.pi/ cc.c['TCMBmuK']**2.
    tsz_cib = np.abs(fgs.tSZ_CIB(ls,testFreq,testFreq)/ls/(ls+1.)*2.*np.pi/ cc.c['TCMBmuK']**2.)

    print ls[print_ells],tsz[print_ells],tsz_cib[print_ells],tsz[print_ells]/tsz_cib[print_ells] 

    totCl = cc.theory.lCl('TT',ls)+ksz+radio+cibp+cibc+noise
    oldtotCl = cc.theory.lCl('TT',ls)+noise
    
    pl = Plotter(scaleY='log')
    pl.add(ls,cc.theory.uCl('TT',ls)*ls**2.,alpha=0.3,ls="--")
    pl.add(ls,cc.theory.lCl('TT',ls)*ls**2.)
    pl.add(ls,noise*ls**2.,label="noise "+str(noiseT)+"uK'")
    pl.add(ls,ksz*ls**2.,label="ksz",alpha=0.2,ls="--")
    pl.add(ls,tsz*ls**2.,label="tsz",alpha=0.2,ls="--")
    pl.add(ls,tsz_cib*ls**2.,label="tsz-cib",alpha=0.2,ls="--")
    # pl.add(ls,kszAlt*ls**2.,label="ksz",alpha=0.5,ls="--")
    pl.add(ls,radio*ls**2.,label="radio",alpha=0.2,ls="--")
    pl.add(ls,cibp*ls**2.,label="cibp",alpha=0.2,ls="--")
    pl.add(ls,cibc*ls**2.,label="cibc",alpha=0.2,ls="--")
    pl.add(ls,totCl*ls**2.,label="total")
    pl.add(ls,oldtotCl*ls**2.,label="total w/o fg",alpha=0.7,ls="--")
    pl.legendOn(loc='lower left',labsize=10)
    pl.done(outDir+"cltt_test"+str(testFreq)+".png")
        




