import matplotlib
matplotlib.use('Agg')
import itertools
from ConfigParser import SafeConfigParser 
import cPickle as pickle
import numpy as np
import sys
from orphics.tools.io import dictFromSection, listFromConfig
from orphics.tools.io import Plotter

expName = sys.argv[1]
calName = sys.argv[2]

iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)

suffix = Config.get('general','suffix')
saveId = expName + "_" + calName + "_" + suffix



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

ms = listFromConfig(Config,calName,'mexprange')
mgrid = np.arange(ms[0],ms[1],ms[2])
zs = listFromConfig(Config,calName,'zrange')
zgrid = np.arange(zs[0],zs[1],zs[2])

dm = np.diff(10**mgrid)
dz = np.diff(zgrid)

paramList = Config.get('fisher','paramList').split(',')
numParams = len(paramList)
Fisher = np.zeros((numParams,numParams))
paramCombs = itertools.combinations_with_replacement(paramList,2)

N_fid = np.load("data/N_dzmq_"+saveId+"_fid"+".npy")
Fellnoq = np.trapz(N_fid,qbins,axis=2)
Fellnom = np.trapz(Fellnoq.T,10**mgrid,axis=1)
pl = Plotter()
pl.add(zgrid,Fellnom)
pl.done("output/nz.png")
N = np.trapz(Fellnom.T,zgrid)
print "Total number of clusters: ", N

for param1,param2 in paramCombs:
    if param1=='tau' or param2=='tau': continue
    if param1=='As' or param2=='As': continue
    dN1 = np.load("data/dN_dzmq_"+saveId+"_"+param1+".npy")
    dN2 = np.load("data/dN_dzmq_"+saveId+"_"+param2+".npy")
    i = paramList.index(param1)
    j = paramList.index(param2)

    assert not(np.any(np.isnan(dN1)))
    assert not(np.any(np.isnan(dN2)))
    assert not(np.any(np.isnan(N_fid)))


    FellBlock = dN1*dN2*np.nan_to_num(1./N_fid)
    Fellnoq = np.trapz(FellBlock,dx=dq,axis=2)
    Fellnoz = np.trapz(Fellnoq,dx=dz,axis=1)
    Fell = np.trapz(Fellnoz,dx=dm)
    
       
    Fisher[i,j] = Fell
    Fisher[j,i] = Fell    


pl = Plotter()
pl.plot2d(Fisher)
pl.done("output/fisher.png")
#print Fisher.shape
print Fisher
Finv = np.linalg.inv(Fisher)
print np.sqrt(np.diagonal(Finv))

