import sys
import matplotlib
matplotlib.use('Agg')
import camb
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import sys, os, time
from szar.counts import ClusterCosmology,SZ_Cluster_Model,Halo_MF

from orphics.tools.io import Plotter,dictFromSection,listFromConfig
from ConfigParser import SafeConfigParser 

import flipper.liteMap as lm

from orphics.tools.cmb import loadTheorySpectraFromCAMB
from alhazen.quadraticEstimator import NlGenerator,getMax


from scipy.fftpack import fftshift,ifftshift,fftfreq
from pyfftw.interfaces.scipy_fftpack import fft2
from pyfftw.interfaces.scipy_fftpack import ifft2

from alhazen.halos import NFWkappa,getDLnMCMB,predictSN
from alhazen.halos import NFWMatchedFilterSN



z = float(sys.argv[1])
    

#saveId = "AdvACTCMBLensingWhiteNoise150GhzTTOnly_MF_N1"
saveId = "ACTPolS1516"

# nsigma = 8.

overdensity = 500.
critical = True
atClusterZ = True


# N = 1000
# numSims = 30

concentration = 1.18
arcStamp = 30.
pxStamp = 0.1
arc_upto = 10.


clusterParams = 'LACluster' # from ini file
cosmologyName = 'LACosmology' # from ini file

iniFile = "input/params.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)

lmax = 8000

cosmoDict = dictFromSection(Config,cosmologyName)
constDict = dictFromSection(Config,'constants')
clusterDict = dictFromSection(Config,clusterParams)
cc = ClusterCosmology(cosmoDict,constDict,lmax)
theory = cc.theory

cambRoot = "data/ell28k_highacc"
gradCut = 2000
halo = True


beamX = 5.0
beamY = 1.5
noiseTX = 45.2
noisePX = np.sqrt(2.)*noiseTX
noiseTY = 16.4
noisePY = np.sqrt(2.)*noiseTY
tellmin = 200
tellmax = 6000
gradCut = 2000
pellmin = 300
pellmax = 6000
polComb = 'TT'


#ls2,Nls2 = np.loadtxt("../alhazen/data/bigDump/ell28_gradCut_2000_polComb_EB_beamY_3.0_noiseY_0.8_grad_sameGrad_tellminY_200_pellminY_50_kmin_40_deg_10.0_px_0.2_delens_1.0.txt",unpack=True)

#ls,Nls = np.loadtxt("data/LA_pol_Nl.txt",unpack=True,delimiter=",")
kmax = 8000

# beamX = 1.5
# beamY = 1.5
# noiseTX = 6.9
# noisePX = np.sqrt(2.)*noiseTX
# noiseTY = 6.9
# noisePY = np.sqrt(2.)*noiseTY
# tellmin = 300
# tellmax = 3000
# gradCut = 2000
# pellmin = 300
# pellmax = 5000
# polComb = 'EB'


kmin = 100


kmax = getMax(polComb,tellmax,pellmax)



# Make a CMB Noise Curve
deg = 10.
px = 0.5
dell = 10
bin_edges = np.arange(kmin,kmax,dell)+dell
theory = loadTheorySpectraFromCAMB(cambRoot,unlensedEqualsLensed=False,useTotal=False,lpad=9000)
lmap = lm.makeEmptyCEATemplate(raSizeDeg=deg, decSizeDeg=deg,pixScaleXarcmin=px,pixScaleYarcmin=px)
myNls = NlGenerator(lmap,theory,bin_edges,gradCut=gradCut)
myNls.updateNoise(beamX,noiseTX,noisePX,tellmin,tellmax,pellmin,pellmax,beamY=beamY,noiseTY=noiseTY,noisePY=noisePY)
ls,Nls = myNls.getNl(polComb=polComb,halo=halo)

ellkk = np.arange(2,9000,1)
Clkk = theory.gCl("kk",ellkk)    
pl = Plotter(scaleY='log',scaleX='log')
pl.add(ellkk,4.*Clkk/2./np.pi)
pl.add(ls,4.*Nls/2./np.pi)
# pl.add(ls2,4.*Nls2/2./np.pi,ls="--")
pl.legendOn(loc='lower left',labsize=10)
pl.done("output/"+saveId+"nl.png")
# sys.exit()

#bin_width = beamY
Clkk = theory.gCl("kk",ls)
Nls = Nls + Clkk


Mexps = np.arange(13.5,15.5,0.05)+0.05
#Mexps = np.arange(14.0,15.7,0.1)
#Mexps = np.arange(14.05,15.75,0.1)

for log10Moverh in Mexps:
    MM = 10.**log10Moverh

    #expectedSN = predictSN(polComb,noiseTY,noisePY,N,MM)
    #print "Rough S/N ", expectedSN

    #dlndm = getDLnMCMB(ls,Nls,cc,log10Moverh,z,concentration,arcStamp,pxStamp,arc_upto,bin_width=beamY,expectedSN=expectedSN,Nclusters=N,numSims=numSims,saveId=saveId,numPoints=1000,nsigma=nsigma,overdensity=overdensity,critical=critical,atClusterZ=atClusterZ)

    sn = NFWMatchedFilterSN(cc,log10Moverh,concentration,z,ells=ls,Nls=Nls,kellmax=kmax,overdensity=overdensity,critical=critical,atClusterZ=atClusterZ,saveId=saveId)


    print log10Moverh, "S/N " , sn


