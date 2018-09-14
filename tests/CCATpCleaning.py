from __future__ import print_function
from __future__ import division
from past.utils import old_div
import matplotlib
matplotlib.use('Agg')
import numpy as np
from szar.counts import ClusterCosmology
from szar.szproperties import SZ_Cluster_Model
from szar.foregrounds import fgNoises, f_nu
from szar.ilc import ILC_simple
import sys,os
from configparser import SafeConfigParser 
import pickle as pickle
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

cf = 1
constraint_tag = ['','_constrained']

#experimentName = "CMB-Probe-50cm"
experimentName = "CCATP-MSIP"
#experimentName = "CCATP-SO-MSIP"
beams = listFromConfig(Config,experimentName,'beams')
noises = listFromConfig(Config,experimentName,'noises')
freqs = listFromConfig(Config,experimentName,'freqs')
lmax = int(Config.getfloat(experimentName,'lmax'))
lknee = listFromConfig(Config,experimentName,'lknee')[0]
alpha = listFromConfig(Config,experimentName,'alpha')[0]
fsky = Config.getfloat(experimentName,'fsky')

SZProfExample = SZ_Cluster_Model(clusterCosmology=cc,clusterDict=clusterDict,rms_noises = noises,fwhms=beams,freqs=freqs,lmax=lmax,lknee=lknee,alpha=alpha)

ILC  = ILC_simple(clusterCosmology=cc, rms_noises = noises,fwhms=beams,freqs=freqs,lmax=lmax,lknee=lknee,alpha=alpha)

experimentName = "CCATP-SO-MSIP"
beams = listFromConfig(Config,experimentName,'beams')
noises = listFromConfig(Config,experimentName,'noises')
freqs = listFromConfig(Config,experimentName,'freqs')
lmax = int(Config.getfloat(experimentName,'lmax'))
lknee = listFromConfig(Config,experimentName,'lknee')[0]
alpha = listFromConfig(Config,experimentName,'alpha')[0]
fsky = Config.getfloat(experimentName,'fsky')

SZProfExample = SZ_Cluster_Model(clusterCosmology=cc,clusterDict=clusterDict,rms_noises = noises,fwhms=beams,freqs=freqs,lmax=lmax,lknee=lknee,alpha=alpha)

ILC2 = ILC_simple(clusterCosmology=cc, rms_noises = noises[:20],fwhms=beams[:20],freqs=freqs[:20],lmax=lmax,lknee=lknee,alpha=alpha)
#ILC3 = ILC_simple(clusterCosmology=cc, rms_noises = noises[:19],fwhms=beams[:19],freqs=freqs[:19],lmax=lmax,lknee=lknee,alpha=alpha)
#ILC4 = ILC_simple(clusterCosmology=cc, rms_noises = noises[:18],fwhms=beams[:18],freqs=freqs[:18],lmax=lmax,lknee=lknee,alpha=alpha)

#5,4,3 CCATp

print((freqs[:3]))
lsedges = np.arange(100,8001,50)

if (cf == 0):

    el_il,  cls_il,  err_il,  s2ny  = ILC.Forecast_Cellyy(lsedges,fsky)
    el_il2, cls_il2, err_il2, s2ny2 = ILC2.Forecast_Cellyy(lsedges,fsky)
#    el_il3, cls_il3, err_il3, s2ny3 = ILC3.Forecast_Cellyy(lsedges,fsky)
#    el_il4, cls_il4, err_il4, s2ny4 = ILC4.Forecast_Cellyy(lsedges,fsky)
    
    el_ilc,  cls_ilc,  err_ilc,  s2n  = ILC.Forecast_Cellcmb(lsedges,fsky)
    el_ilc2, cls_ilc2, err_ilc2, s2n2 = ILC2.Forecast_Cellcmb(lsedges,fsky)
#    el_ilc3, cls_ilc3, err_ilc3, s2n3 = ILC3.Forecast_Cellcmb(lsedges,fsky)
#    el_ilc4, cls_ilc4, err_ilc4, s2n4 = ILC4.Forecast_Cellcmb(lsedges,fsky)

if (cf == 1):
    el_il,  cls_il,  err_il,  s2ny  = ILC.Forecast_Cellyy(lsedges,fsky,constraint="cmb")
    el_il2, cls_il2, err_il2, s2ny2 = ILC2.Forecast_Cellyy(lsedges,fsky,constraint="cmb")
#    el_il3, cls_il3, err_il3, s2ny3 = ILC3.Forecast_Cellyy(lsedges,fsky,constraint="cmb")
#    el_il4, cls_il4, err_il4, s2ny4 = ILC4.Forecast_Cellyy(lsedges,fsky,constraint="cmb")
    
    el_ilc,  cls_ilc,  err_ilc,  s2n  = ILC.Forecast_Cellcmb(lsedges,fsky,constraint="tsz")
    el_ilc2, cls_ilc2, err_ilc2, s2n2 = ILC2.Forecast_Cellcmb(lsedges,fsky,constraint="tsz")
#    el_ilc3, cls_ilc3, err_ilc3, s2n3 = ILC3.Forecast_Cellcmb(lsedges,fsky,constraint="tsz")
#    el_ilc4, cls_ilc4, err_ilc4, s2n4 = ILC4.Forecast_Cellcmb(lsedges,fsky,constraint="tsz")

print(('S/N y', s2ny, s2ny2))#,s2ny3, s2ny4))

print(('S/N CMB', s2n, s2n2))#, s2n3, s2n4))

#print 'S/N' , np.sqrt(np.sum((cls_ilc/err_ilc)**2))

outDir = "/Users/nab/Desktop/Projects/SO_forecasts/"

outfile1 = outDir + experimentName + "_y_weights"+constraint_tag[cf]+".png"
outfile2 = outDir + experimentName + "_cmb_weights"+constraint_tag[cf]+".png"

ILC.PlotyWeights(outfile1)
ILC.PlotcmbWeights(outfile2)

if (cf == 0): 
    eln,nl = ILC.Noise_ellyy()
    eln2,nl2 = ILC2.Noise_ellyy()
#    eln3,nl3 = ILC3.Noise_ellyy()
#    eln4,nl4 = ILC4.Noise_ellyy()
    
    elnc,nlc = ILC.Noise_ellcmb()
    elnc2,nlc2 = ILC2.Noise_ellcmb()
#    elnc3,nlc3 = ILC3.Noise_ellcmb()
#    elnc4,nlc4 = ILC4.Noise_ellcmb()

if (cf == 1): 
    eln,nl = ILC.Noise_ellyy(constraint='cib')
    eln2,nl2 = ILC2.Noise_ellyy(constraint='cib')
#    eln3,nl3 = ILC3.Noise_ellyy(constraint='cib')
#    eln4,nl4 = ILC4.Noise_ellyy(constraint='cib')
    
    elnc,nlc = ILC.Noise_ellcmb(constraint='tsz')
    elnc2,nlc2 = ILC2.Noise_ellcmb(constraint='tsz')
#    elnc3,nlc3 = ILC3.Noise_ellcmb(constraint='tsz')
#    elnc4,nlc4 = ILC4.Noise_ellcmb(constraint='tsz')

pl = Plotter(labelX="$\ell$",labelY="Noise Ratio",ftsize=12,figsize=(8,6))
pl.add(eln2,old_div(nl2,nl),label="SO/CCATP")
#pl.add(eln3,nl3/nl,label="90 - 270 / Full")
#pl.add(eln4,nl4/nl,label="90 - 220 / Full")
#pl.legend(loc='upper right',labsize=10)
pl.done(outDir+experimentName+"_y_noise_ratio"+constraint_tag[cf]+".png")

pl = Plotter(labelX="$\ell$",labelY="Noise Ratio",ftsize=12,figsize=(8,6))
pl.add(elnc2,old_div(nlc2,nlc),label="SO/CCATP")
#pl.add(elnc3,nlc3/nlc,label="90 - 270 / Full")
#pl.add(elnc4,nlc4/nlc,label="90 - 220 / Full")
#pl.legend(loc='upper right',labsize=10)
pl.done(outDir+experimentName+"_cmb_noise_ratio"+constraint_tag[cf]+".png")

pl = Plotter(labelX="$\ell$",labelY="Error Ratio",ftsize=12,figsize=(8,6))
pl.add(el_il2,old_div(err_il2,err_il),label="SO/CCATP")
#pl.add(el_il3,err_il3/err_il,label="90 - 270 / Full")
#pl.add(el_il4,err_il4/err_il,label="90 - 220 / Full")
#pl.legend(loc='upper right',labsize=10)
pl.done(outDir+experimentName+"_y_error_ratio"+constraint_tag[cf]+".png")

pl = Plotter(labelX="$\ell$",labelY="Noise Ratio",ftsize=12,figsize=(8,6))
pl.add(el_ilc2,old_div(err_ilc2,err_ilc),label="SO/CCATP")
#pl.add(el_ilc3,err_ilc3/err_ilc,label="90 - 270 / Full")
#pl.add(el_ilc4,err_ilc4/err_ilc,label="90 - 220 / Full")
#pl.legend(loc='upper right',labsize=10)
pl.done(outDir+experimentName+"_cmb_error_ratio"+constraint_tag[cf]+".png")

ellfac = el_ilc*(el_ilc + 1.) / (2.*np.pi) * 1e12 * constDict['TCMB']**2
ellfac2 = eln*(eln + 1.) / (2.*np.pi) * 1e12 * constDict['TCMB']**2

pl = Plotter(labelX="$\ell$",labelY="$C_\ell \, (1 + \ell) \ell / 2\pi \, [\mu \mathrm{K}]$",ftsize=12,figsize=(8,6),scaleY='log')
pl._ax.set_ylim(1,10000)
pl._ax.set_xlim(100.,5000.)
pl.add(el_ilc,cls_ilc*ellfac,color='black')
pl.add(eln,nl*ellfac2,label="$N_\ell$ CCATP")
pl.add(eln2,nl2*ellfac2,label="$N_\ell$ SO")
pl.addErr(el_ilc,cls_ilc*ellfac,err_ilc*ellfac,label="CCATP")
pl.addErr(el_ilc2+10,cls_ilc2*ellfac,err_ilc2*ellfac,label="SO")
#pl.legend(loc='upper right',labsize=10)
pl.done(outDir+experimentName+"_cmb_cls"+constraint_tag[cf]+".png")
ls = np.arange(2,8000,10)

pl = Plotter(labelX="$\ell$",labelY="$C_\ell \, (1 + \ell) \ell / 2\pi \, [\mu \mathrm{K}]$",ftsize=12,figsize=(8,6),scaleY='log')
pl._ax.set_ylim(0.1,1000)
pl._ax.set_xlim(100.,8000.)
pl.add(el_il,cls_il*ellfac,color='black')
pl.add(elnc,nlc*ellfac2,label="$N_\ell$ CCATP")
pl.add(elnc2,nlc2*ellfac2,label="$N_\ell$ SO")
pl.addErr(el_il,cls_il*ellfac,err_il*ellfac,label="CCATP")
pl.addErr(el_il2+10,cls_il2*ellfac,err_il2*ellfac,label="SO")
pl.legendOn(loc='upper right',labsize=10)
pl.done(outDir+experimentName+"_y_cls"+constraint_tag[cf]+".png")
ls = np.arange(2,8000,10)

#ksz = fgs.ksz_temp(ls)/ls/(ls+1.)*2.*np.pi/ cc.c['TCMBmuK']**2.


        




