import matplotlib
matplotlib.use('Agg')
import itertools
from ConfigParser import SafeConfigParser 
import cPickle as pickle
import numpy as np
import sys
from orphics.tools.io import dictFromSection, listFromConfig
from orphics.tools.io import Plotter
import matplotlib.pyplot as plt

from szlib.szcounts import getTotN


expName = sys.argv[1]
gridName = sys.argv[2]
calName = sys.argv[3]


iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)
bigDataDir = Config.get('general','bigDataDirectory')

zmax = Config.getfloat('general','zmax')
fsky = Config.getfloat(expName,'fsky')

suffix = Config.get('general','suffix')
saveId = expName + "_" + gridName + "_" + calName + "_" + suffix

# get s/n q-bins
qs = listFromConfig(Config,'general','qbins')
qspacing = Config.get('general','qbins_spacing')
if qspacing=="log":
    qbins = np.logspace(np.log10(qs[0]),np.log10(qs[1]),int(qs[2]))
elif qspacing=="linear":
    qbins = np.linspace(qs[0],qs[1],int(qs[2]))
else:
    raise ValueError
dq = np.diff(qbins)


# get mass and z grids
ms = listFromConfig(Config,gridName,'mexprange')
mgrid = np.arange(ms[0],ms[1],ms[2])
zs = listFromConfig(Config,gridName,'zrange')
zgrid = np.arange(zs[0],zs[1],zs[2])
zgrid = zgrid[zgrid<zmax]
zlen = zgrid.size
dm = np.diff(10**mgrid)
dz = np.diff(zgrid)


# Fiducial number counts
N_fid = np.load(bigDataDir+"N_dzmq_"+saveId+"_fid"+".npy")
N_fid = N_fid[:,:zlen,:]*fsky
print "Total number of clusters: ", getTotN(N_fid,mgrid,zgrid,qbins)

massGridName = bigDataDir+"lensgrid_"+expName+"_"+gridName+"_"+calName+".pkl"
mgrid,zgrid,Merrgrid = pickle.load(open(massGridName,'rb'))

nnoq = np.trapz(N_fid,qbins,axis=2)
nnoq = nnoq[:-1,:]*dm
nnoq = nnoq[:,:-1]*dz
mperbin = Merrgrid[:,:zlen] *100./ np.sqrt(nnoq)

pl = Plotter()
pl.plot2d(mperbin)
pl.done("output/mperbin.png")

