import orphics.tools.cmb as cmb
from ConfigParser import SafeConfigParser 
from orphics.tools.io import dictFromSection, listFromConfig
import orphics.tools.io as io
from orphics.theory.cosmology import Cosmology
import os,sys
import numpy as np

iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)

freq_to_use = 150.

cc = Cosmology(lmax=8000,pickling=True)
theory = cc.theory

ells = np.arange(2,8000,1)
cltt = theory.lCl('TT',ells)
TCMB = 2.7255e6

out_dir = os.environ['WWW']

pl = io.Plotter(scaleY='log',labelX="$\ell$",labelY="$\ell^2 C_{\ell}$",ftsize=20)

for expName,lab in zip(['S4-1.0-0.4-noatm','S4-1.0-0.4'],['S4 1arc no atm','S4 1arc lknee=5500 (in paper)']):

    beam = listFromConfig(Config,expName,'beams')
    noise = listFromConfig(Config,expName,'noises')
    freq = listFromConfig(Config,expName,'freqs')
    lkneeT,lkneeP = listFromConfig(Config,expName,'lknee')
    alphaT,alphaP = listFromConfig(Config,expName,'alpha')

    print expName,beam,noise,lkneeT,lkneeP,alphaT,alphaP
    

    ind = np.where(np.isclose(freq,freq_to_use))
    beamFind = np.array(beam)[ind]
    noiseFind = np.array(noise)[ind]

    nls = cmb.noise_func(ells,beamFind,noiseFind,lknee=lkneeT,alpha=alphaT,TCMB=TCMB)


    
    pl.add(ells,nls*ells**2.,label=lab)


lkneeT = 1000.
nls = cmb.noise_func(ells,beamFind,noiseFind,lknee=lkneeT,alpha=alphaT,TCMB=TCMB)
pl.add(ells,nls*ells**2.,label='S4 1arc lknee=1000',alpha=0.5,ls="--")
lkneeT = 3000.
nls = cmb.noise_func(ells,beamFind,noiseFind,lknee=lkneeT,alpha=alphaT,TCMB=TCMB)
pl.add(ells,nls*ells**2.,label='S4 1arc lknee=3000',alpha=0.5,ls="--")
    

pl.add(ells,cltt*ells**2.,color="k")
pl._ax.set_xlim(100,8000)
pl.legendOn(labsize=12,loc='upper right')
pl.done(out_dir+"testatm.png")


ttlknee = np.array([350.,3400.,4900.])
size = np.array([0.5,5.,7.]) # size in meters

freq = 150.e9
cspeed = 299792458.
wavelength = cspeed/freq
resin = 1.22*wavelength/size*60.*180./np.pi



lknees = []
beamList = np.arange(1.0,20.0,0.1)
for beam in beamList:

    lkneet, alphat, lkneep, alphap = cmb.getAtmosphere(beamFWHMArcmin=beam,returnFunctions=False)
    lknees.append(lkneet)


pl=io.Plotter(labelX="FWHM arcminutes",labelY="lknee_T",ftsize=14)
pl.add(beamList,lknees)
pl.add(resin,ttlknee,ls="none",marker="o",label="M.Hasselfield fits")
pl.legendOn(labsize=14,loc="upper right")
pl.done(out_dir+"lknees.png")
