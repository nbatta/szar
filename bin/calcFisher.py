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
#Fellnoq = np.trapz(N_fid,qbins,axis=2)
Fellnoq = np.sum(N_fid[:,:,1:]*dq,axis=2)

print dm.shape
#Fellnom = np.trapz(Fellnoq.T,10**mgrid,axis=1)
Fellnom = np.sum(Fellnoq[1:,:].T*dm,axis=1)
pl = Plotter()
pl.add(zgrid,Fellnom)
pl.done("output/nz.png")
#N = np.trapz(Fellnom.T,zgrid)
N = np.sum(Fellnom[1:]*dz)
print N
#sys.exit()

for param1,param2 in paramCombs:
    if param1=='tau' or param2=='tau': continue
    dN1 = np.load("data/dN_dzmq_"+saveId+"_"+param1+".npy")
    dN2 = np.load("data/dN_dzmq_"+saveId+"_"+param2+".npy")
    i = paramList.index(param1)
    j = paramList.index(param2)

    FellBlock = np.nan_to_num(dN1*dN2/N_fid)
    Fellnoq = np.trapz(FellBlock,dx=dq,axis=2)
    Fellnoz = np.trapz(Fellnoq,dx=dz,axis=1)
    Fell = np.trapz(Fellnoz,dx=dm)
    
    # if param1=='H0':
    #     pl = Plotter()
    #     pl.plot2d(dN1[:,:,63])
    #     pl.done("output/dslice.png")

       
    Fisher[i,j] = Fell
    Fisher[j,i] = Fell    


print Fisher.shape
print Fisher
# Finv = np.linalg.inv(Fisher)
# print np.sqrt(np.diagonal(Finv))

