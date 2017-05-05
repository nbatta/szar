import matplotlib
matplotlib.use('Agg')
import numpy as np
from szar.counts import ClusterCosmology,Halo_MF,getNmzq
from szar.szproperties import SZ_Cluster_Model

import sys,os
from ConfigParser import SafeConfigParser 
import cPickle as pickle

from orphics.tools.io import dictFromSection, listFromConfig,dictOfListsFromSection, Plotter

maxSteps = None
outDir = os.environ['WWW']+"plots/steps/"

def getCents(xs):
    return (xs[1:]+xs[:-1])/2.

expName = sys.argv[1]
gridName = sys.argv[2]
calName = sys.argv[3]
paramName = sys.argv[4]


iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)
version = Config.get('general','version')
bigDataDir = Config.get('general','bigDataDirectory')
pzcut = Config.get('general','photoZCutOff')

version = Config.get('general','version')
calFile = bigDataDir+"lensgrid_"+expName+"_"+gridName+"_"+calName+ "_v" + version+".pkl"
massMultiplier = Config.getfloat('general','mass_calib_factor')

# load the mass calibration grid
mexp_edges, z_edges, lndM = pickle.load(open(calFile,"rb"))

mgrid,zgrid,siggrid = pickle.load(open(bigDataDir+"szgrid_"+expName+"_"+gridName+ "_v" + version+".pkl",'rb'))
assert np.all(mgrid==mexp_edges)
assert np.all(z_edges==zgrid)

saveId = expName + "_" + gridName + "_" + calName + "_v" + version


stepdict = dictOfListsFromSection(Config,'steps')
steptests = {}
fparams = {}   
for (key, val) in Config.items('params'):
    if ',' in val:
        param, step = val.split(',')
        if key=='sigR':
            rayFid = float(param)
            rayStep = float(step)
        fparams[key] = float(param)
    else:
        fparams[key] = float(val)



constDict = dictFromSection(Config,'constants')
clusterDict = dictFromSection(Config,'cluster_params')
beam = listFromConfig(Config,expName,'beams')
noise = listFromConfig(Config,expName,'noises')
freq = listFromConfig(Config,expName,'freqs')
lknee = listFromConfig(Config,expName,'lknee')[0]
alpha = listFromConfig(Config,expName,'alpha')[0]

clttfile = Config.get('general','clttfile')

# get s/n q-bins
qs = listFromConfig(Config,'general','qbins')
qspacing = Config.get('general','qbins_spacing')
if qspacing=="log":
    qbin_edges = np.logspace(np.log10(qs[0]),np.log10(qs[1]),int(qs[2])+1)
elif qspacing=="linear":
    qbin_edges = np.linspace(qs[0],qs[1],int(qs[2])+1)
else:
    raise ValueError
        
        
yNzs = {}
key = paramName
#for key in stepdict.keys():
if True:

    yNzs[key] = []
    vals = stepdict[key][:maxSteps]
    vals.sort()
    for val in vals:
        print key, val
        
        uppassparams = fparams.copy()
        dnpassparams = fparams.copy()

        uppassparams[key] = fparams[key]+val/2.
        dnpassparams[key] = fparams[key]-val/2.


        cc = ClusterCosmology(uppassparams,constDict,clTTFixFile=clttfile)
        HMF = Halo_MF(cc,mexp_edges,z_edges)
        HMF.sigN = siggrid.copy()
        SZProf = SZ_Cluster_Model(cc,clusterDict,rms_noises = noise,fwhms=beam,freqs=freq,lknee=lknee,alpha=alpha)
        Nup = HMF.N_of_mqz_SZ(lndM*massMultiplier,qbin_edges,SZProf)


        cc = ClusterCosmology(dnpassparams,constDict,clTTFixFile=clttfile)
        HMF = Halo_MF(cc,mexp_edges,z_edges)
        HMF.sigN = siggrid.copy()
        SZProf = SZ_Cluster_Model(cc,clusterDict,rms_noises = noise,fwhms=beam,freqs=freq,lknee=lknee,alpha=alpha)
        Ndn = HMF.N_of_mqz_SZ(lndM*massMultiplier,qbin_edges,SZProf)
        
    
        dNdp = (getNmzq(Nup,mexp_edges,z_edges,qbin_edges)-getNmzq(Ndn,mexp_edges,z_edges,qbin_edges))/val


        Nz = dNdp.copy().sum(axis=-1).sum(axis=0)
        Nm = dNdp.copy().sum(axis=-1).sum(axis=-1)
        Nq = dNdp.copy().sum(axis=0).sum(axis=0)
        
        yNzs[key].append((val,Nz,Nm,Nq))
        

        
    pl = Plotter(labelX="$z$",labelY="$dN$")
    xstep = 0.01
    for i,val in enumerate(vals):
        assert yNzs[key][i][0]==val
        pl.add(getCents(z_edges)+((i-len(vals)/2)*xstep),yNzs[key][i][1],label=key+" "+str(val))
    pl.legendOn(labsize=10,loc='upper right')
    pl.done(outDir+key+"_Nz_step.png")
    pl = Plotter(labelX="$M$",labelY="$dN$")
    xstep = 0.01
    for i,val in enumerate(vals):
        assert yNzs[key][i][0]==val
        pl.add(getCents(mexp_edges)+((i-len(vals)/2)*xstep),yNzs[key][i][2],label=key+" "+str(val))
    pl.legendOn(labsize=10,loc='upper right')
    pl.done(outDir+key+"_Nm_step.png")
    pl = Plotter(labelX="$q$",labelY="$dN$",scaleX='log')
    xstep = 0.1
    for i,val in enumerate(vals):
        assert yNzs[key][i][0]==val
        pl.add(getCents(qbin_edges)+((i-len(vals)/2)*xstep),yNzs[key][i][3],label=key+" "+str(val))
    pl.legendOn(labsize=10,loc='upper right')
    pl.done(outDir+key+"_Nq_step.png")

