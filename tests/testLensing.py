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








cambRoot = "data/ell28k_highacc"
gradCut = None
halo = False
beam = 7.0
noise = 27.0
tellmin = 2
tellmax = 3000

pellmin = 2
pellmax = 3000

deg = 10.
px = 0.5

bin_edges = np.arange(10,3000,10)

theory = loadTheorySpectraFromCAMB(cambRoot,unlensedEqualsLensed=False,useTotal=False,lpad=9000)
arc = deg*60.
lmap = lm.makeEmptyCEATemplate(raSizeDeg=arc/60., decSizeDeg=arc/60.,pixScaleXarcmin=px,pixScaleYarcmin=px)
print lmap.data.shape
myNls = NlGenerator(lmap,theory,bin_edges,gradCut=None)

myNls.updateNoise(beam,noise,tellmin,tellmax,pellmin,pellmax)

polCombList = ['TT','EE','ET','TE','EB','TB']
colorList = ['red','blue','green','cyan','orange','purple']
ellkk = np.arange(2,9000,1)
Clkk = theory.gCl("kk",ellkk)    


pl = Plotter(scaleY='log',scaleX='log')
pl.add(ellkk,4.*Clkk/2./np.pi)

# CHECK THAT NORM MATCHES HU/OK
for polComb,col in zip(polCombList,colorList):
    ls,Nls = myNls.getNl(polComb='TT')
    try:
        huFile = '/astro/u/msyriac/repos/cmb-lensing-projections/data/NoiseCurvesKK/hu_'+polComb.lower()+'.csv'
        huell,hunl = np.loadtxt(huFile,unpack=True,delimiter=',')
    except:
        huFile = '/astro/u/msyriac/repos/cmb-lensing-projections/data/NoiseCurvesKK/hu_'+polComb[::-1].lower()+'.csv'
        huell,hunl = np.loadtxt(huFile,unpack=True,delimiter=',')


    pl.add(ls,4.*Nls/2./np.pi,color=col)
    pl.add(huell,hunl,ls='--',color=col)


pl.done("tests/output/testbin.png")





sys.exit()









from szlib.lensing import kappa

print kappa(cc,MM,3.2,zz,1.0)


arc = 20.
px = 0.01


lmap = lm.makeEmptyCEATemplate(raSizeDeg=arc/60., decSizeDeg=arc/60.,pixScaleXarcmin=px,pixScaleYarcmin=px)
from szlib.lensing import GRFGen




#@timeit
def quickStamp(size,px,ell,Cell):
    lmap = lm.makeEmptyCEATemplate(raSizeDeg=size/60., decSizeDeg=size/60.,pixScaleXarcmin=px,pixScaleYarcmin=px)
    lmap.fillWithGaussianRandomField(ell,Cell,bufferFactor = 1)
    return lmap.data

xMap,yMap,modRMap,xx,xy = fmaps.getRealAttributes(lmap)
lxMap,lyMap,modLMap,thetaMap,lx,ly = fmaps.getFTAttributesFromLiteMap(lmap)

kappaMap = kappa(cc,MM,3.2,zz,modRMap*180.*60./np.pi)
kappaMap = fmaps.stepFunctionFilterLiteMap(kappaMap,modLMap,6000)

ell = cc.ells
Cell = cc.cltt

cmb = quickStamp(arc,px,ell,Cell)



from szlib.lensing import GRFGen

generator = GRFGen(lmap,ell,Cell)
#cmb = quickStamp(arc,px,ell,Cell)
#cmb2 = generator.getMap()



@timeit
def testOld(N,arc,px,ell,Cell):
    for i in xrange(N):
        cmb = quickStamp(arc,px,ell,Cell)


@timeit
def testNew(N,gen):
    for i in xrange(N):
        cmb = gen.getMap()


testOld(50,arc,px,ell,Cell)
testNew(50,generator)
    

pl = Plotter()
pl.plot2d(cmb)
pl.done("output/cmb.png")

pl = Plotter()
pl.plot2d(cmb2)
pl.done("output/cmb2.png")


pl = Plotter()
pl.plot2d(modRMap)
pl.done("output/rmap.png")

pl = Plotter()
pl.plot2d(kappaMap)
pl.done("output/kmap.png")

