import matplotlib
matplotlib.use('Agg')
import camb
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import sys, os, time
from szlib.szcounts import ClusterCosmology,SZ_Cluster_Model,Halo_MF,dictFromSection,listFromConfig,timeit

from orphics.tools.output import Plotter
from ConfigParser import SafeConfigParser 


#def quickNl(beam,noise,



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

from szlib.lensing import kappa

print kappa(cc,MM,3.2,zz,1.0)

import liteMap as lm
import orphics.analysis.flatMaps as fmaps

arc = 20.
px = 0.01


lmap = lm.makeEmptyCEATemplate(raSizeDeg=arc/60., decSizeDeg=arc/60.,pixScaleXarcmin=px,pixScaleYarcmin=px)
from szlib.lensing import GRFGen
ell = cc.ells
Cell = cc.cltt

N = 2048
lmap = lm.getEmptyMapWithDifferentDims(lmap,N,N)
generator = GRFGen(lmap,ell,Cell)
cmb = generator.getMap()
pl = Plotter()
pl.plot2d(cmb)
pl.done("output/cmbForNick.png")
sys.exit()



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

