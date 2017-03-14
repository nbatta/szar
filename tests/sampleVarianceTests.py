import matplotlib
matplotlib.use('Agg')
import sys
import numpy as np
from szlib.szcounts import ClusterCosmology,SZ_Cluster_Model,Halo_MF,sampleVarianceOverNsquareOverBsquare,haloBias,getTotN
from orphics.tools.io import Plotter,dictFromSection,listFromConfig
from ConfigParser import SafeConfigParser 

clusterParams = 'LACluster' # from ini file
cosmologyName = 'LACosmology' # from ini file
iniFile = "input/params.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)

lmax = 3000
cosmoDict = dictFromSection(Config,cosmologyName)
constDict = dictFromSection(Config,'constants')
clusterDict = dictFromSection(Config,clusterParams)
cc = ClusterCosmology(cosmoDict,constDict,skipCls=True)


# mrange = np.arange(13.5,15.71,0.3)
# zrange = np.arange(0.05,3.0,0.3)

mrange = np.arange(13.5,15.71,0.05)
zrange = np.arange(0.05,3.0,0.1)

# mrange = np.arange(13.5,15.71,0.02)
# zrange = np.arange(0.02,3.0,0.02)

fsky=0.4

hmf = Halo_MF(cc,mrange,zrange)

hb = haloBias(mrange,zrange,cc.rhoc0om,hmf.kh,hmf.pk)
pl = Plotter()
pl.plot2d(hb)
pl.done("output/hb.png")

# ls = "-"
# lab = ""
# pl = Plotter(labelX="$z$",labelY="b",ftsize=14)
# pl.add(zrange,hb[np.where(np.isclose(mrange,14.0)),:].ravel(),ls=ls,label=lab+" 10^14 Msol/h")
# pl.add(zrange,hb[np.where(np.isclose(mrange,14.3)),:].ravel(),ls=ls,label=lab+" 10^14.3 Msol/h")
# pl.add(zrange,hb[np.where(np.isclose(mrange,14.5)),:].ravel(),ls=ls,label=lab+" 10^14.5 Msol/h")
# pl.add(zrange,hb[np.where(np.isclose(mrange,14.7)),:].ravel(),ls=ls,label=lab+" 10^14.7 Msol/h")
# pl.legendOn(loc='upper right',labsize=8)
# pl.done("output/slicebias.png")

# sys.exit()

powers = sampleVarianceOverNsquareOverBsquare(cc,hmf.kh,hmf.pk,mrange,zrange,fsky,lmax=1000)
sovernsquare = hb*hb*powers

qs = [6.,500.,64]
qbins = np.logspace(np.log10(qs[0]),np.log10(qs[1]),int(qs[2]))


# iniFile = "input/pipeline.ini"
# Config = SafeConfigParser()
# Config.optionxform=str
# Config.read(iniFile)
# qs = listFromConfig(Config,'general','qbins')
# qspacing = Config.get('general','qbins_spacing')
# if qspacing=="log":
#     qbins = np.logspace(np.log10(qs[0]),np.log10(qs[1]),int(qs[2]))
# elif qspacing=="linear":
#     qbins = np.linspace(qs[0],qs[1],int(qs[2]))
# else:
#     raise ValueError


# Nfile = "/astro/astronfs01/workarea/msyriac/data/SZruns/master/sigN.txt"
# sgrid = np.loadtxt(Nfile)
# pl = Plotter()
# pl.plot2d(sgrid)
# pl.done("output/sgridfineMaster.png")

Nfile = "/astro/astronfs01/workarea/msyriac/data/SZruns/refactor/szgrid_S4-5m_grid-default.pkl"
import cPickle as pickle
md,zd,sgrid = pickle.load(open(Nfile,'rb'))
print sgrid.shape
pl = Plotter()
pl.plot2d(sgrid)
pl.done("output/sgridfineRefactor.png")




# Nfile = "/astro/astronfs01/workarea/msyriac/data/SZruns/master/N_dzmq_S4-5m_CMB_all_coarse_master_test_fid.npy"
# n = np.load(Nfile)
# print getTotN(n[:,:,:],mrange[:],zrange,qbins,returnNz=False)*fsky
# nnoq = np.trapz(n,qbins,axis=2)*fsky
# pl = Plotter()
# pl.plot2d(nnoq)
# pl.done("output/ngridfineMaster.png")

outDir = "/gpfs01/astro/www/msyriac/"
Nfile = "/astro/astronfs01/workarea/msyriac/data/SZruns/refactor/N_dzmq_S4-5m_grid-default_CMB_all_refactor_test_fid.npy"
n = np.load(Nfile)
print n.shape
print getTotN(n,mrange,zrange,qbins,returnNz=False)*fsky
nnoq = np.trapz(n,qbins,axis=2)*fsky
pl = Plotter()
pl.plot2d(nnoq)
pl.done(outDir+"ngridfineRefactor.png")

# sys.exit()


ms = 10**mrange
dms = np.diff(ms)
dM = np.outer(dms,np.ones([len(zrange)]))

dZ = np.diff(zrange)



nnoq = nnoq[:-1,:]*dM
nnoq = nnoq[:,:-1]*dZ

print nnoq.shape

outDir = "/gpfs01/astro/www/msyriac/"
pl = Plotter()
pl.plot2d((nnoq))
pl.done(outDir+ "poisson.png")


pl = Plotter()
pl.plot2d((sovernsquare[:-1,:-1]*nnoq*nnoq))
pl.done(outDir+ "sample.png")

pl = Plotter()
pl.plot2d((sovernsquare[:-1,:-1]*nnoq))
pl.done(outDir+ "ratio.png")
