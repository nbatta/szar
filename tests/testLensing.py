import sys
import matplotlib
matplotlib.use('Agg')
import camb
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import sys, os, time
from szlib.szcounts import ClusterCosmology,SZ_Cluster_Model,Halo_MF

from orphics.tools.io import Plotter,dictFromSection,listFromConfig
from ConfigParser import SafeConfigParser 

import flipper.liteMap as lm

from orphics.tools.cmb import loadTheorySpectraFromCAMB
from alhazen.quadraticEstimator import NlGenerator,getMax


from scipy.fftpack import fftshift,ifftshift,fftfreq
from pyfftw.interfaces.scipy_fftpack import fft2
from pyfftw.interfaces.scipy_fftpack import ifft2

from alhazen.halos import NFWkappa,getDLnMCMB,predictSN




saveId = "test"
nsigma = 8.
zz = 0.7
MM = 2.e14
log10Moverh = np.log10(MM)
# overdensity = 180.
# critical = False
# atClusterZ = False

overdensity = 500.
critical = True
atClusterZ = True


N = 1000
numSims = 30

concentration = 3.2
arcStamp = 30.
#pxStamp = 0.2
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


cambRoot = "data/ell28k_highacc"
gradCut = 2000
halo = True


# beamX = 3.0
# beamY = 3.0
# noiseTX = 1.0
# noisePX = 1.414
# noiseTY = 1.0
# noisePY = 1.414
# tellmin = 200
# tellmax = 5000
# gradCut = 2000
# pellmin = 200
# pellmax = 3000
# polComb = 'EE'
# kmin = 100


beamX = 1.0
beamY = 1.0
noiseTX = 10.0
noisePX = 14.14
noiseTY = 10.0
noisePY = 14.14
tellmin = 2
tellmax = 8000
gradCut = 2000
pellmin = 2
pellmax = 8000
polComb = 'TT'
kmin = 100


# beamX = 1.5
# beamY = 1.5
# noiseTX = 12.0
# noisePX = 14.14
# noiseTY = 12.0
# noisePY = 14.14
# tellmin = 2
# tellmax = 6000
# gradCut = 2000
# pellmin = 2
# pellmax = 8000
# polComb = 'TT'
# kmin = 100


# beamX = 7.0
# beamY = 7.0
# noiseTX = 27.0
# noisePX = 14.14
# noiseTY = 27.0
# noisePY = 14.14
# tellmin = 2
# tellmax = 3000
# gradCut = 2000
# pellmin = 2
# pellmax = 3000
# polComb = 'TB'
# kmin = 100


kmax = getMax(polComb,tellmax,pellmax)



expectedSN = predictSN(polComb,noiseTY,noisePY,N,MM)
print "Rough S/N ", expectedSN

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
pl.legendOn(loc='lower left',labsize=10)
pl.done("output/"+saveId+"nl.png")

bin_width = beamY


dlndm = getDLnMCMB(ls,Nls,cc,log10Moverh,zz,concentration,arcStamp,pxStamp,arc_upto,bin_width=beamY,expectedSN=expectedSN,Nclusters=N,numSims=numSims,saveId="test",numPoints=1000,nsigma=nsigma,overdensity=overdensity,critical=critical,atClusterZ=atClusterZ)

print "S/N " , 1./dlndm


