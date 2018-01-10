import os, sys
import numpy as np
from orphics.io import Plotter

from configparser import SafeConfigParser 
iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)
from orphics.io import dictFromSection, listFromConfig
constDict = dictFromSection(Config,'constants')
clttfile = Config.get('general','clttfile')
gridName = "grid-default"
ms = listFromConfig(Config,gridName,'mexprange')
Mexp_edges = np.arange(ms[0],ms[1]+ms[2],ms[2])
zs = listFromConfig(Config,gridName,'zrange')
z_edges = np.arange(zs[0],zs[1]+zs[2],zs[2])


zrange = (z_edges[1:]+z_edges[:-1])/2.
#np.arange(0.05,3.05,0.1)

cambRoot = os.environ['HOME']+"/software/CAMB_wa/"

pl = Plotter(scaleX='log')#,scaleY='log')

#stepList = ['0.1','0.05','0.2','1.0','0.001']
#colList = ['C0','C1','C2','C3','C4']

stepList = ['2.0','1.5','1.0','0.2']
colList = ['C0','C1','C2','C3']

# stepList = ['0.1','0.05','0.2','0.001']
# colList = ['C0','C1','C2','C4']

for step,col in zip(stepList,colList):
    dRoot = cambRoot+"forDerivsStep"+step
    for z in zrange:

        khup,Pup = np.loadtxt(dRoot+"Up_matterpower_"+str(z)+".dat",unpack=True)
        khdn,Pdn = np.loadtxt(dRoot+"Dn_matterpower_"+str(z)+".dat",unpack=True)
        assert np.all(np.isclose(khup,khdn))
        stepF = float(step)
        dP = (Pup-Pdn)/stepF

        #if z<2.5: continue
        if z>0.2: continue

        #alph = 1.-z/3.0
        alph = 1.
        pl.add(khup,-dP,color=col,alpha=alph)
    pl.add(khup,-dP*0.,color=col,alpha=1.,ls="-",label=step)

pl.legendOn()
pl.done(os.environ['WWW']+"dps.png")




fparams = {}   # the 
for (key, val) in Config.items('params'):
    if ',' in val:
        param, step = val.split(',')
        fparams[key] = float(param)
    else:
        fparams[key] = float(val)
passParams = fparams
from szar.counts import ClusterCosmology,Halo_MF
cc = ClusterCosmology(passParams,constDict,clTTFixFile=clttfile)
HMF = Halo_MF(cc,Mexp_edges,z_edges)
kh = HMF.kh
pk = HMF.pk

pl = Plotter(scaleX='log')#,scaleY='log')
from scipy.interpolate import interp1d

for i,z in enumerate(zrange[:]):

    khfid,pkfid = np.loadtxt(cambRoot+"forDerivsFid_matterpower_"+str(z)+".dat",unpack=True)

    
    # pl.add(khfid,pkfid,alpha=1.-z/3.0,color="C0")
    # pl.add(kh,pk[i],alpha=1.-z/3.0,color="C1")

    pkfidI = interp1d(khfid,pkfid,bounds_error=False,fill_value=np.nan)(kh)

    pl.add(kh,(pk[i]-pkfidI)*100./pk[i],alpha=1.-z/3.0,color="C1")
    
    
# pl.legendOn()
pl.done(os.environ['WWW']+"ps.png")
