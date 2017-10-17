import matplotlib
matplotlib.use('Agg')
import numpy as np
from szar.counts import ClusterCosmology
from szar.szproperties import SZ_Cluster_Model
from szar.foregrounds import fgNoises, f_nu
from szar.ilc import ILC_simple
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


experimentName = "SO-v2-6m"
beams = listFromConfig(Config,experimentName,'beams')
noises = listFromConfig(Config,experimentName,'noises')
freqs = listFromConfig(Config,experimentName,'freqs')
lmax = int(Config.getfloat(experimentName,'lmax'))
lknee = listFromConfig(Config,experimentName,'lknee')[0]
alpha = listFromConfig(Config,experimentName,'alpha')[0]
fsky = Config.getfloat(experimentName,'fsky')

SZProfExample = SZ_Cluster_Model(clusterCosmology=cc,clusterDict=clusterDict,rms_noises = noises,fwhms=beams,freqs=freqs,lmax=lmax,lknee=lknee,alpha=alpha,tsz_cib=True)

#print SZProfExample.nl / SZProfExample.nl_new

pl = Plotter()
pl.add(SZProfExample.evalells,SZProfExample.nl_old / SZProfExample.nl)
pl.done("tests/new_nl_test.png")

#ILC = ILC_simple(clusterCosmology=cc, rms_noises = noises,fwhms=beams,freqs=freqs,lmax=lmax,lknee=lknee,alpha=alpha)
#ILC2 = ILC_simple(clusterCosmology=cc, rms_noises = noises[3:],fwhms=beams[3:],freqs=freqs[3:],lmax=lmax,lknee=lknee,alpha=alpha)
#ILC3 = ILC_simple(clusterCosmology=cc, rms_noises = noises[3:6],fwhms=beams[3:6],freqs=freqs[3:6],lmax=lmax,lknee=lknee,alpha=alpha)

ILC = ILC_simple(clusterCosmology=cc, rms_noises = noises,fwhms=beams,freqs=freqs,lmax=lmax,lknee=lknee,alpha=alpha)

lsedges = np.arange(300,8001,100)
el_ilc, cls_ilc, err_ilc, s2n = ILC.Forecast_Cellcmb(lsedges,fsky)
el_ilc_c, cls_ilc_c, err_ilc_c, s2n_c = ILC.Forecast_Cellcmb(lsedges,fsky,constraint='tsz')
print s2n,s2n_c

ILC = ILC_simple(clusterCosmology=cc, rms_noises = noises,fwhms=beams,freqs=freqs,lmax=7000,lknee=lknee,alpha=alpha)

lsedges = np.arange(300,7001,100)
el_ilc, cls_ilc, err_ilc, s2n = ILC.Forecast_Cellcmb(lsedges,fsky)
print s2n

ILC = ILC_simple(clusterCosmology=cc, rms_noises = noises,fwhms=beams,freqs=freqs,lmax=6000,lknee=lknee,alpha=alpha)

lsedges = np.arange(300,6001,100)
el_ilc, cls_ilc, err_ilc, s2n = ILC.Forecast_Cellcmb(lsedges,fsky)
print s2n

ILC = ILC_simple(clusterCosmology=cc, rms_noises = noises,fwhms=beams,freqs=freqs,lmax=5000,lknee=lknee,alpha=alpha)

lsedges = np.arange(300,5001,100)
el_ilc, cls_ilc, err_ilc, s2n = ILC.Forecast_Cellcmb(lsedges,fsky)
print s2n

ILC = ILC_simple(clusterCosmology=cc, rms_noises = noises,fwhms=beams,freqs=freqs,lmax=3000,lknee=lknee,alpha=alpha)

lsedges = np.arange(300,3001,100)
el_ilc, cls_ilc, err_ilc, s2n = ILC.Forecast_Cellcmb(lsedges,fsky)
print s2n

ILC = ILC_simple(clusterCosmology=cc, rms_noises = noises,fwhms=beams,freqs=freqs,lmax=2000,lknee=lknee,alpha=alpha)

lsedges = np.arange(300,2001,100)
el_ilc, cls_ilc, err_ilc, s2n = ILC.Forecast_Cellcmb(lsedges,fsky)
print s2n

#print 'S/N' , np.sqrt(np.sum((cls_ilc/err_ilc)**2))

outDir = "tests/"

outfile1 = outDir + experimentName + "_y_weights.png"
outfile2 = outDir + experimentName + "_cmb_weights.png"

ILC.PlotyWeights(outfile1)
ILC.PlotcmbWeights(outfile2)

eln,nl = ILC.Noise_ellyy()
#eln2,nl2 = ILC2.Noise_ellyy()
#eln3,nl3 = ILC3.Noise_ellyy()

elnc,nlc = ILC.Noise_ellcmb()
#elnc2,nlc2 = ILC2.Noise_ellcmb()
#elnc3,nlc3 = ILC3.Noise_ellcmb()

#pl = Plotter()
##pl.add(eln,nl*eln**2,label="Full")
#pl.add(eln2,nl2/nl,label="90 - 270 / Full")
#pl.add(eln3,nl3/nl,label="90 - 220 / Full")
#pl.legendOn(loc='upper left',labsize=10)
#pl.done(outDir+"noise_test.png")

#pl = Plotter()
#pl.add(elnc2,nlc2/nlc,label="90 - 270 / Full")
#pl.add(elnc3,nlc3/nlc,label="90 - 220 / Full")
#pl.legendOn(loc='upper left',labsize=10)
#pl.done(outDir+"noise_test_CMB.png")

#outDir = os.environ['WWW']+"plots/"

ls = np.arange(2,8000,10)

ksz = fgs.ksz_temp(ls)/ls/(ls+1.)*2.*np.pi/ cc.c['TCMBmuK']**2.
# kszAlt = fgs.ksz_battaglia_test(ls)/ls/(ls+1.)*2.*np.pi/ cc.c['TCMBmuK']**2.

print_ells = [100,200,300,400,500,600]

print "freqs", freqs
print "freqs", freqs[3:]
print "freqs", freqs[3:6]

fq_mat   = np.matlib.repmat(freqs,len(freqs),1)
fq_mat_t = np.transpose(np.matlib.repmat(freqs,len(freqs),1))

#f_nu_arr2 = np.array(freqs)*0.0
#for ii in xrange(len(freqs)):
#    f_nu_arr2[ii] = f_nu_old(cc.c,freqs[ii])

f_nu_arr = f_nu(cc.c,np.array(freqs))

#print "TEST", np.sum(f_nu_arr - f_nu_arr2)

#print fq_mat
#print fq_mat_t

radio_mat = fgs.rad_ps(print_ells[4],fq_mat,fq_mat_t) / cc.c['TCMBmuK']**2.

#print "contraction", np.dot(np.transpose(f_nu_arr),np.dot(np.linalg.inv(radio_mat),f_nu_arr))

#print fgs.rad_ps(ls[10],fq_mat_t,fq_mat)/ls[10]/(ls[10]+1.)*2.*np.pi/ cc.c['TCMBmuK']**2.

#print fgs.rad_ps(ls[10],fq_mat_t,fq_mat)*0.0 + 1.

print "noise", noise_func(print_ells[4],np.array(beams),np.array(noises),lknee,alpha,dimensionless=False) / cc.c['TCMBmuK']**2.

fac_norm = ls*(ls+1.)/(2.*np.pi) * cc.c['TCMBmuK']**2

for fwhm,noiseT,testFreq in zip(beams,noises,freqs):
    totCl = 0.
    #print testFreq
    noise = noise_func(ls,fwhm,noiseT,lknee,alpha,dimensionless=False) / cc.c['TCMBmuK']**2.
    
    radio = fgs.rad_ps(ls,testFreq,testFreq)/ls/(ls+1.)*2.*np.pi/ cc.c['TCMBmuK']**2.
    cibp = fgs.cib_p(ls,testFreq,testFreq) /ls/(ls+1.)*2.*np.pi/ cc.c['TCMBmuK']**2.
    cibc = fgs.cib_c(ls,testFreq,testFreq)/ls/(ls+1.)*2.*np.pi/ cc.c['TCMBmuK']**2.
    tsz = fgs.tSZ(ls,testFreq,testFreq)/ls/(ls+1.)*2.*np.pi/ cc.c['TCMBmuK']**2.
    tsz_cib = np.abs(fgs.tSZ_CIB(ls,testFreq,testFreq)/ls/(ls+1.)*2.*np.pi/ cc.c['TCMBmuK']**2.)

    pol_dust = fgs.gal_dust_pol(ls,testFreq,testFreq) /ls/(ls+1.)*2.*np.pi/ cc.c['TCMBmuK']**2.
    pol_ps   = fgs.rad_pol_ps(ls,testFreq,testFreq) /ls/(ls+1.)*2.*np.pi/ cc.c['TCMBmuK']**2.
    pol_sync = fgs.gal_sync_pol(ls,testFreq,testFreq) /ls/(ls+1.)*2.*np.pi/ cc.c['TCMBmuK']**2.

    print ls[print_ells],tsz[print_ells],tsz_cib[print_ells],tsz[print_ells]/tsz_cib[print_ells] 


    totCl = cc.theory.lCl('TT',ls)+ksz+radio+cibp+cibc+noise+tsz+tsz_cib
    oldtotCl = cc.theory.lCl('TT',ls)+noise
    
    pl = Plotter(scaleY='log')
    pl._ax.set_ylim(1,10000)
    pl._ax.set_xlim(100.,9000.)
    pl.add(ls,cc.theory.uCl('TT',ls)*ls**2.,alpha=0.3,ls="--")
    pl.add(ls,cc.theory.lCl('TT',ls)*ls**2.)
    pl.add(ls,noise*fac_norm,label="noise "+str(noiseT)+"uK'")
    pl.add(ls,ksz*fac_norm,label="ksz",alpha=0.2,ls="--")
    pl.add(ls,tsz*fac_norm,label="tsz",alpha=0.2,ls="--")
    pl.add(ls,tsz_cib*fac_norm,label="tsz-cib",alpha=0.2,ls="--")
    # pl.add(ls,kszAlt*ls**2.,label="ksz",alpha=0.5,ls="--")
    pl.add(ls,radio*fac_norm,label="radio",alpha=0.2,ls="--")
    pl.add(ls,cibp*fac_norm,label="cibp",alpha=0.2,ls="--")
    pl.add(ls,cibc*fac_norm,label="cibc",alpha=0.2,ls="--")
    pl.add(ls,totCl*fac_norm,label="total")
    pl.add(ls,oldtotCl*fac_norm,label="total w/o fg",alpha=0.7,ls="--")
    pl.legendOn(loc='lower left',labsize=10)
    pl.done(outDir+"cltt_test"+str(testFreq)+".png")
        
    pl = Plotter(scaleY='log')
    pl._ax.set_ylim(1,1000)
    pl._ax.set_xlim(30.,5000.)
    totClEE = cc.theory.lCl('EE',ls)+pol_dust+pol_sync+pol_ps

    print ls[print_ells],pol_ps[print_ells],pol_sync[print_ells],pol_dust[print_ells],totClEE[print_ells]

    pl.add(ls,cc.theory.lCl('EE',ls)*fac_norm,ls="--")
    pl.add(ls,pol_dust*fac_norm,ls="--")
    pl.add(ls,pol_sync*fac_norm)
    pl.add(ls,pol_ps*fac_norm)
    pl.add(ls,totClEE*fac_norm)
    pl.done(outDir+"clee_test"+str(testFreq)+".png")



