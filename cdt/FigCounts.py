from __future__ import print_function
from __future__ import division
from builtins import zip
from past.utils import old_div
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from configparser import SafeConfigParser
from orphics.tools.io import Plotter
import pickle as pickle
import sys

from szar.counts import ClusterCosmology,Halo_MF,getNmzq
from szar.szproperties import SZ_Cluster_Model
import numpy as np

#from orphics.analysis.flatMaps import interpolateGrid



out_dir = "/gpfs01/astro/www/msyriac/paper/"

iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)
bigDataDir = Config.get('general','bigDataDirectory')
clttfile = Config.get('general','clttfile')

gridName = "grid-default"
version = "0.5"


from orphics.tools.io import dictFromSection, listFromConfig
constDict = dictFromSection(Config,'constants')
clusterDict = dictFromSection(Config,'cluster_params')

fparams = {} 
for (key, val) in Config.items('params'):
    if ',' in val:
        param, step = val.split(',')
        fparams[key] = float(param)
    else:
        fparams[key] = float(val)


cc = ClusterCosmology(fparams,constDict,clTTFixFile=clttfile)

from matplotlib.patches import Rectangle


expList = ['S4-highres','S4-lowres']
labList = ['S4 1.0\' 0.95uK\'','S4 1.5\' 0.85uK\'']
pad = 0.05

pl = Plotter(labelX="$z$",labelY="$N(z)$",ftsize=12,scaleY='log',figsize=(6,4))
#pl = Plotter(labelX="$z$",labelY="$N(z)$",ftsize=12)

colList = ['C0','C1','C2','C3','C4']
Ndict = {}
for expName,col,labres in zip(expList,colList,labList):

    mgrid,zgrid,siggrid = pickle.load(open(bigDataDir+"szgrid_"+expName+"_"+gridName+ "_v" + version+".pkl",'rb'))
    
    z_edges = zgrid
    zrange = old_div((z_edges[1:]+z_edges[:-1]),2.)
    mexp_edges = mgrid

    beam = listFromConfig(Config,expName,'beams')
    noise = listFromConfig(Config,expName,'noises')
    freq = listFromConfig(Config,expName,'freqs')
    lknee = listFromConfig(Config,expName,'lknee')[0]
    alpha = listFromConfig(Config,expName,'alpha')[0]
    fsky = Config.getfloat(expName,'fsky')
    HMF = Halo_MF(cc,mexp_edges,z_edges)
    HMF.sigN = siggrid.copy()

    
    SZProf = SZ_Cluster_Model(cc,clusterDict,rms_noises = noise,fwhms=beam,freqs=freq,lknee=lknee,alpha=alpha)
    Nofzs = np.multiply(HMF.N_of_z_SZ(fsky,SZProf),np.diff(z_edges).reshape(1,z_edges.size-1)).ravel()
    print((Nofzs.sum()))

    currentAxis = plt.gca()

    zbins = np.array([0.,1.0,2.0,3.0])#z_edges #np.arange(0.,3.5,0.5)
    k = 0
    Ndict[labres] = []
    for zleft,zright in zip(zbins[:-1],zbins[1:]):
        zcent = old_div((zleft+zright),2.)
        xerr = old_div((zright-zleft),2.)
        N = Nofzs[np.logical_and(zrange>zleft,zrange<=zright)].sum()
        print((zleft,zright,N))
        if k==0:
            lab = labres
        else:
            lab = None
            
        
        currentAxis.add_patch(Rectangle((zcent - xerr+pad, 0), 2*xerr-old_div(pad,2.), N, facecolor=col,label=lab))#,alpha=0.5))
        Ndict[labres].append(N)
        k+=1
    
    

pl.legendOn(labsize=9,loc='upper right')
pl._ax.set_ylim(1,5.e4) 
pl._ax.set_xlim(0.,3.)
#pl.done(out_dir+"cdtCounts.png")



