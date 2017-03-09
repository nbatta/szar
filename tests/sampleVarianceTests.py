import matplotlib
matplotlib.use('Agg')
import sys
import numpy as np
from szlib.szcounts import ClusterCosmology,SZ_Cluster_Model,Halo_MF,sampleVarianceOverNsquareOverBsquare,haloBias
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

mrange = np.arange(13.5,15.7,0.05)
zrange = np.arange(0.05,3.0,0.1)

fsky=0.4

hmf = Halo_MF(cc,mrange,zrange)

hb = haloBias(mrange,zrange,cc.rhoc0om,hmf.kh,hmf.pk)
# pl = Plotter()
# pl.plot2d(hb)
# pl.done("output/hb.png")

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

powers = sampleVarianceOverNsquareOverBsquare(cc,hmf.kh,hmf.pk,mrange,zrange,fsky,lmax=2000)

sovernsquare = hb*hb*powers

Nfile = "data/N_dzmq_S4-7m_CMB_all_wstep_fid.npy"
n = np.load(Nfile)
print n.shape
qs = [6.,500.,64]
qbins = np.logspace(np.log10(qs[0]),np.log10(qs[1]),int(qs[2]))

nnoq = np.trapz(n,qbins,axis=2)*fsky

pl = Plotter()
pl.plot2d(nnoq)
pl.done("output/ns.png")


ms = 10**mrange
dms = np.diff(ms)
dM = np.outer(dms,np.ones([len(zrange)]))

dZ = np.diff(zrange)



nnoq = nnoq[:-1,:]*dM
nnoq = nnoq[:,:-1]*dZ

print nnoq.shape




pl = Plotter()
pl.plot2d(sovernsquare[:-1,:-1]*nnoq)
pl.done("output/sovernsquare.png")