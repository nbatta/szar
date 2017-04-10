import matplotlib
matplotlib.use('Agg')
import camb
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import sys, os, time
from szlib.szcounts import ClusterCosmology,SZ_Cluster_Model,Halo_MF,getTotN
from orphics.tools.io import Plotter,dictFromSection,listFromConfig
from ConfigParser import SafeConfigParser 
import cPickle as pickle
from orphics.tools.io import Plotter
from orphics.analysis.flatMaps import interpolateGrid

clusterParams = 'cluster_params' # from ini file
cosmologyName = 'params' # from ini file

iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)
bigDataDir = Config.get('general','bigDataDirectory')
version = Config.get('general','version')

outDir=os.environ['WWW']

experimentName = sys.argv[1]
gridName = sys.argv[2]
calName = sys.argv[3]
cal = calName
exp = experimentName
expName = experimentName

beam = listFromConfig(Config,experimentName,'beams')
noise = listFromConfig(Config,experimentName,'noises')
freq = listFromConfig(Config,experimentName,'freqs')
lmax = int(Config.getfloat(experimentName,'lmax'))
lknee = listFromConfig(Config,experimentName,'lknee')[0]
alpha = listFromConfig(Config,experimentName,'alpha')[0]
fsky = Config.getfloat(experimentName,'fsky')


cosmoDict = dictFromSection(Config,cosmologyName)
constDict = dictFromSection(Config,'constants')
clusterDict = dictFromSection(Config,clusterParams)
cc = ClusterCosmology(cosmoDict,constDict,clTTFixFile = "data/cltt_lensed_Feb18.txt")#,skipCls=True)


if "owl" in calName:
    calFile = bigDataDir+"lensgrid_grid-"+cal+"_"+cal+".pkl"
else:
    calFile = bigDataDir+"lensgrid_"+exp+"_"+gridName+"_"+calName+ "_v" + version+".pkl"


Mexp_edges, z_edges, lndM = pickle.load(open(calFile,"rb"))
mgrid,zgrid,siggrid = pickle.load(open(bigDataDir+"szgrid_"+expName+"_"+gridName+ "_v" + version+".pkl",'rb'))
zs = (z_edges[1:]+z_edges[:-1])/2.


hmf = Halo_MF(cc,Mexp_edges,z_edges)


SZProf = SZ_Cluster_Model(cc,clusterDict,rms_noises = noise,fwhms=beam,freqs=freq,lknee=lknee,alpha=alpha)


N1 = hmf.N_of_z()*fsky
hmf.sigN = siggrid
N2 = hmf.N_of_z_SZ(SZProf)*fsky

    
pl = Plotter(scaleY='log')
pl.add(zs,N1)
pl.add(zs,N2)

Ntot0 = np.dot(N1,np.diff(z_edges))
Ntot1 = np.dot(N2,np.diff(z_edges))
print "All clusters in the Universe  ",Ntot0
print "All clusters detectable at qmin ",SZProf.qmin," is ",Ntot1


sn,ntot = hmf.Mass_err(fsky,lndM,SZProf)
outmerr = lndM

print "All clusters according to Mass_err ", ntot

# get s/n q-bins
qs = listFromConfig(Config,'general','qbins')
qspacing = Config.get('general','qbins_spacing')
if qspacing=="log":
    qbin_edges = np.logspace(np.log10(qs[0]),np.log10(qs[1]),int(qs[2])+1)
elif qspacing=="linear":
    qbin_edges = np.linspace(qs[0],qs[1],int(qs[2])+1)
else:
    raise ValueError

q_arr = (qbin_edges[1:]+qbin_edges[:-1])/2.

dnqmz = hmf.N_of_mqz_SZ(outmerr,qbin_edges,SZProf)

N,Nofz = getTotN(dnqmz,Mexp_edges,z_edges,qbin_edges,returnNz=True)

print "All clusters according to dnqmz ",N*fsky
print "All clusters according to \dzdnqmz ",np.dot(Nofz,np.diff(z_edges))*fsky

pl.add(zs,Nofz*fsky,label="mqz")
pl.legendOn()
pl.done(outDir+"ncompare.png")


# nnoq = np.dot(dnqmz,np.diff(qbin_edges))*fsky
# pl = Plotter()
# pl.plot2d(nnoq)
# pl.done(outDir+"ncompare2.png")
