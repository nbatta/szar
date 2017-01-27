import sys
import matplotlib
matplotlib.use('Agg')
import camb
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import sys, os, time
from szlib.szcounts import ClusterCosmology,SZ_Cluster_Model,Halo_MF,dictFromSection,listFromConfig

from orphics.tools.output import Plotter
from ConfigParser import SafeConfigParser 

import flipper.liteMap as lm

from orphics.tools.cmb import loadTheorySpectraFromCAMB
from alhazen.quadraticEstimator import NlGenerator,getMax


from scipy.fftpack import fftshift,ifftshift,fftfreq
from pyfftw.interfaces.scipy_fftpack import fft2
from pyfftw.interfaces.scipy_fftpack import ifft2

from alhazen.halos import NFWkappa,getDLnMCMB,predictSN



z = float(sys.argv[1])
    

saveId = "AdvACTCMBLensingWhiteNoise150GhzTTOnly"

nsigma = 8.

overdensity = 500.
critical = True
atClusterZ = True


N = 1000
numSims = 30

concentration = 3.2
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


cambRoot = "data/ell28k_highacc"
gradCut = 2000
halo = True


beamX = 1.5
beamY = 1.5
noiseTX = 6.9
noisePX = np.sqrt(2.)*noiseTX
noiseTY = 6.9
noisePY = np.sqrt(2.)*noiseTY
tellmin = 300
tellmax = 3000
gradCut = 2000
pellmin = 300
pellmax = 5000
polComb = 'TT'
kmin = 100


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


Mexps = np.arange(14.0,15.7,0.1)

for log10Moverh in Ms:
    dlndm = getDLnMCMB(ls,Nls,cc,log10Moverh,z,concentration,arcStamp,pxStamp,arc_upto,bin_width=beamY,expectedSN=expectedSN,Nclusters=N,numSims=numSims,saveId=saveId,numPoints=1000,nsigma=nsigma,overdensity=overdensity,critical=critical,atClusterZ=atClusterZ)

    print "S/N " , 1./dlndm


