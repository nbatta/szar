import szar.sims as s
import orphics.tools.io as io
from configparser import SafeConfigParser
from enlib import enmap,utils,lensing,powspec
import os, sys
import numpy as np
from flipper.fft import fft as fft_gen,ifft as ifft_gen
import orphics.tools.stats as stats
from alhazen.halos import NFWkappa
from szar.counts import ClusterCosmology

fake_kappa = False


iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)
constDict = io.dictFromSection(Config,'constants')


sim = s.BattagliaSims(constDict,rootPath="/global/cscratch1/sd/msyriac/ClusterSims/")
cc = sim.cc #ClusterCosmology(constDict=constDict,lmax=6000,pickling=True)
#sim.mapReader(plotRel=True)


# === TEMPLATE MAP ===
px = 0.2
arc = 30
hwidth = arc/2.
deg = utils.degree
arcmin =  utils.arcmin
shape, wcs = enmap.geometry(pos=[[-hwidth*arcmin,-hwidth*arcmin],[hwidth*arcmin,hwidth*arcmin]], res=px*arcmin, proj="car")


snap = 46
Nmax = 300
mThreshold = 10.**(13.6)

kappaStack = 0.
szStack = 0.

modlmap = enmap.modlmap(shape,wcs)
pix_ells = np.arange(0,modlmap.max(),1)
modr_map = enmap.modrmap(shape,wcs) * 180.*60./np.pi
bin_edges = np.arange(0.,10.0,0.5)
binner = stats.bin2D(modr_map,bin_edges)



if fake_kappa:
    # === NFW CLUSTER ===
    massOverh = 2e14*cc.h
    zL = 0.5
    sourceZ = 1100.
    overdensity = 500.
    critical = True
    atClusterZ = True
    concentration = 3.0
    
    
    comS = cc.results.comoving_radial_distance(cc.cmbZ)*cc.h
    comL = cc.results.comoving_radial_distance(zL)*cc.h
    winAtLens = (comS-comL)/comS
    nfwMap_fake,r500 = NFWkappa(cc,massOverh,concentration,zL,modr_map,winAtLens,overdensity=overdensity,critical=critical,atClusterZ=atClusterZ)


    ps_noise = np.zeros((3,3,pix_ells.size))
    ps_noise[0,0] = pix_ells*0.+0.000000001
    ps_noise[1,1] = pix_ells*0.+0.000000001
    ps_noise[2,2] = pix_ells*0.+0.000000001





kprofiles = []
sprofiles = []
jprofiles = []

N = 0
avgMass = 0.
for massIndex in range(Nmax):
    print(massIndex)
    if sim.trueM500[str(snap)][massIndex]<mThreshold: continue
    kappaMap,szMap,projectedM500,z = sim.getKappaSZ(snap,massIndex,shape,wcs,apodWidthArcmin=2.0)


    if fake_kappa:
        noise = enmap.rand_map(shape,wcs,ps_noise,scalar=True)
        kappaMap = nfwMap_fake + noise



    cents, kprofile = binner.bin(kappaMap)
    cents, sprofile = binner.bin(szMap)
    kprofiles.append(kprofile)
    sprofiles.append(sprofile)
    jprofiles.append(np.append(kprofile.ravel(),sprofile.ravel()))
    kappaStack += kappaMap
    szStack += szMap
    N+=1
    avgMass += sim.trueM500[str(snap)][massIndex]

print(N)
    
kappaStack /= N
szStack /= N
avgMass /= N
    
print("done")

out_dir = os.environ['WWW']+"plots/"

io.quickPlot2d(kappaStack,out_dir+"kappa.png")#,crange=[-0.1,0.1])
io.quickPlot2d(szStack,out_dir+"sz.png")#,crange=[-50,0])


kstats = stats.getStats(kprofiles)
sstats = stats.getStats(sprofiles)
jstats = stats.getStats(jprofiles)

io.quickPlot2d(stats.cov2corr(jstats['cov']),out_dir+"jcorr.png")
io.quickPlot2d(stats.cov2corr(kstats['cov']),out_dir+"kcorr.png")
io.quickPlot2d(stats.cov2corr(sstats['cov']),out_dir+"scorr.png")

cents, mkprofile = binner.bin(kappaStack)
cents, msprofile = binner.bin(szStack)




pl = io.Plotter()
pl.addErr(cents,sstats['mean'],yerr=sstats['err'],ls="-")
#pl.add(cents,msprofile,ls="--")
pl.done(out_dir+"sprofile.png")


# === NFW CLUSTER ===
massOverh = avgMass*cc.h
zL = sim.snapToZ(snap)
sourceZ = 1100.
overdensity = 500.
critical = True
atClusterZ = True
concentration = cc.Mdel_to_cdel(massOverh,zL,overdensity)
print(("Duffy Concentration : ", concentration))


print((massOverh, zL, concentration))
comS = cc.results.comoving_radial_distance(cc.cmbZ)*cc.h
comL = cc.results.comoving_radial_distance(zL)*cc.h
winAtLens = (comS-comL)/comS
nfwMap,r500 = NFWkappa(cc,massOverh,concentration,zL,modr_map,winAtLens,overdensity=overdensity,critical=critical,atClusterZ=atClusterZ)

io.quickPlot2d(modr_map,out_dir+"modrmap.png")
io.quickPlot2d(nfwMap,out_dir+"nfwmap.png")

cents, nkprofile = binner.bin(nfwMap)

pl = io.Plotter()#scaleY='log',scaleX='log')
pl.addErr(cents,kstats['mean'],yerr=kstats['err'],ls="-")
pl.add(cents,nkprofile,ls="--")




# # === NFW CLUSTER ===
# massOverh = 2.e14
# zL = 0.7
# overdensity = 180.
# critical = False
# atClusterZ = False
# concentration = 3.2
# comS = cc.results.comoving_radial_distance(cc.cmbZ)*cc.h
# comL = cc.results.comoving_radial_distance(zL)*cc.h
# winAtLens = (comS-comL)/comS
# nfwMap,r500 = NFWkappa(cc,massOverh,concentration,zL,modr_map,winAtLens,overdensity=overdensity,critical=critical,atClusterZ=atClusterZ)
# cents, nkprofile = binner.bin(nfwMap)
# pl.add(cents,nkprofile,ls="-.")


#pl._ax.set_xlim(0.2,10.)
#pl._ax.set_ylim(0.001,2.)
pl.done(out_dir+"kprofile.png")
