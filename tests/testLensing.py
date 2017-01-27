import sys
import matplotlib
matplotlib.use('Agg')
import camb
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import sys, os, time
from szlib.szcounts import ClusterCosmology,SZ_Cluster_Model,Halo_MF,dictFromSection,listFromConfig
from orphics.tools.stats import timeit

from orphics.tools.output import Plotter
from ConfigParser import SafeConfigParser 

import flipper.liteMap as lm
import orphics.analysis.flatMaps as fmaps

from orphics.tools.cmb import loadTheorySpectraFromCAMB
from orphics.theory.quadEstTheory import NlGenerator

from szlib.lensing import GRFGen




zz = 0.5
MM = 5.e14
clusterParams = 'LACluster' # from ini file
cosmologyName = 'LACosmology' # from ini file
experimentName = "AdvAct"

iniFile = "input/params.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)


beam = listFromConfig(Config,experimentName,'beams')
noise = listFromConfig(Config,experimentName,'noises')
freq = listFromConfig(Config,experimentName,'freqs')
lmax = int(Config.getfloat(experimentName,'lmax'))
fsky = Config.getfloat(experimentName,'fsky')



cosmoDict = dictFromSection(Config,cosmologyName)
constDict = dictFromSection(Config,'constants')
clusterDict = dictFromSection(Config,clusterParams)
cc = ClusterCosmology(cosmoDict,constDict,lmax)



ell = cc.ells
Cell = cc.cltt





stepfilter_ellmax = 8000


cambRoot = "data/ell28k_highacc"
gradCut = None
halo = True
beamX = 1.5
beamY = 1.5
noiseT = 1.0
noiseP = 1.414
tellmin = 100
tellmax = 8000
gradCut = 2000

pellmin = 100
pellmax = 8000
polComb = 'EB'

deg = 10.
px = 0.5
arc = deg*60.

bin_edges = np.arange(100,8000,10)

theory = loadTheorySpectraFromCAMB(cambRoot,unlensedEqualsLensed=False,useTotal=False,lpad=9000)
lmap = lm.makeEmptyCEATemplate(raSizeDeg=arc/60., decSizeDeg=arc/60.,pixScaleXarcmin=px,pixScaleYarcmin=px)
print lmap.data.shape
myNls = NlGenerator(lmap,theory,bin_edges,gradCut=gradCut)


myNls.updateNoise(beamX,noiseT,noiseP,tellmin,tellmax,pellmin,pellmax,beamY=beamY)
ls,Nls = myNls.getNl(polComb=polComb,halo=halo)

ellkk = np.arange(2,9000,1)
Clkk = theory.gCl("kk",ellkk)    


pl = Plotter(scaleY='log',scaleX='log')
pl.add(ellkk,4.*Clkk/2./np.pi)

pl.add(ls,4.*Nls/2./np.pi)
pl.legendOn(loc='lower left',labsize=10)
pl.done("output/nl.png")








from szlib.lensing import kappa

print kappa(cc,MM,3.2,zz,1.0)
arc = 30.
px = 0.1
lmap = lm.makeEmptyCEATemplate(raSizeDeg=arc/60., decSizeDeg=arc/60.,pixScaleXarcmin=px,pixScaleYarcmin=px)


xMap,yMap,modRMap,xx,xy = fmaps.getRealAttributes(lmap)
lxMap,lyMap,modLMap,thetaMap,lx,ly = fmaps.getFTAttributesFromLiteMap(lmap)

kappaMap = kappa(cc,MM,3.2,zz,modRMap*180.*60./np.pi)
finetheta = np.arange(0.01,10.,0.01)
finekappa = kappa(cc,MM,3.2,zz,finetheta)
kappaMap = fmaps.stepFunctionFilterLiteMap(kappaMap,modLMap,stepfilter_ellmax)


centers, thprof = binner.bin(kappaMap)


from szlib.lensing import GRFGen
generator = GRFGen(lmap,ls,Nls)

from orphics.tools.stats import bin2D,getStats,fchisq
bin_edges = np.arange(0.,10.,1.0)
binner = bin2D(modRMap*180.*60./np.pi, bin_edges)

N = 100

@timeit
def getProfiles(N):
    profiles = []
    for i in range(N):
        noise = generator.getMap(stepFilterEll=stepfilter_ellmax)
        stamp = kappaMap + noise
        
        centers, profile = binner.bin(stamp)
        
        profiles.append(profile)
    return profiles


pl = Plotter()
pl.plot2d(kappaMap)
pl.done("output/kappa.png")


profiles = getProfiles(N)
stats = getStats(profiles)
pl = Plotter()
pl.add(centers,thprof,lw=2,color='black')
pl.add(finetheta,finekappa,lw=2,color='black',ls="--")
pl.addErr(centers,stats['mean'],yerr=stats['errmean'],lw=2)
pl._ax.set_ylim(-0.01,0.3)
pl.done("output/profile.png")


amplitudeRange = np.arange(-1.,2.,0.01)
width = amplitudeRange[1]-amplitudeRange[0]
amplist = []
print "Fitting amplitudes..."
for prof in profiles:
    Likelihood = lambda x: np.exp(-0.5*fchisq(prof,siginv,thprof,amp=x))
    Likes = np.array([Likelihood(x) for x in amplitudeRange])
    LikeTot += L
Likes = Likes / (Likes.sum()*width) #normalize
    #ampBest,ampErr = cfit(norm.pdf,amplitudeRange,Likes,p0=[1.0,0.5])[0]

    amplist.append(ampBest)



# siginv = np.linalg.pinv(cov[:len(thetas),:len(thetas)])
# #print siginv
# #print radmeans[:len(thetas)]
# b = np.dot(siginv,radmeans[:len(thetas)])
# chisq = np.dot(radmeans[:len(thetas)],b)

# print np.sqrt(chisq*Nclus/Nsupp)
