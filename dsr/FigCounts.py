from __future__ import print_function
from __future__ import division
from builtins import zip
from past.utils import old_div
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from configparser import SafeConfigParser
from orphics.io import Plotter
import pickle as pickle
import sys
from orphics import io

from szar.counts import ClusterCosmology,Halo_MF,getNmzq
from szar.szproperties import SZ_Cluster_Model
import numpy as np



out_dir = "/gpfs01/astro/www/msyriac/plots/"

iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)
bigDataDir = Config.get('general','bigDataDirectory')
clttfile = Config.get('general','clttfile')

gridName = "grid-default"
version = Config.get('general','version')


from orphics.io import dict_from_section, list_from_config as listFromConfig
constDict = dict_from_section(Config,'constants')
clusterDict = dict_from_section(Config,'cluster_params')

#CB_color_cycle = ['#1C110A','#E4D6A7','#E9B44C','#9B2915','#50A2A7']
CB_color_cycle = ['#1C110A','#B561A7','#E4D6A7','#9B2915','#50A2A7']


# import matplotlib as mpl
# mpl.rcParams['axes.color_cycle'] = CB_color_cycle

fparams = {} 
for (key, val) in Config.items('params'):
    if ',' in val:
        param, step = val.split(',')
        fparams[key] = float(param)
    else:
        fparams[key] = float(val)


cc = ClusterCosmology(fparams,constDict,clTTFixFile=clttfile)

from matplotlib.patches import Rectangle


expList = ['SO-v3-base-40','S4-v3style-fsky-40']
labList = ['SO-base','CMB-S4']
pad = 0.05

pl = Plotter(xlabel="$z$",ylabel="$N(z)$",ftsize=20,yscale='log',figsize=(6,4),thk=2,labsize=16)
#pl = Plotter(labelX="$z$",labelY="$N(z)$",ftsize=12)


colList = ['C0','C1']
Ndict = {}
for expName,col,labres in zip(expList,colList,labList):

    mgrid,zgrid,siggrid = pickle.load(open(bigDataDir+"szgrid_"+expName+"_"+gridName+ "_v" + version+".pkl",'rb'),encoding='latin1')
    
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
    # SZProf = SZ_Cluster_Model(cc,clusterDict,rms_noises = [1],fwhms=[1],freqs=[1],lknee=0,alpha=1)
    Nofzs = np.multiply(HMF.N_of_z_SZ(fsky,SZProf),np.diff(z_edges).reshape(1,z_edges.size-1)).ravel()
    # io.save_cols(expName+"_zcent_Ntot.txt",(zrange,Nofzs))
    print(expName,(Nofzs.sum()))

    currentAxis = plt.gca()

    zbins = z_edges #np.arange(0.,3.5,0.5)
    k = 0
    Ndict[labres] = []
    for zleft,zright in zip(zbins[:-1],zbins[1:]):
        zcent = old_div((zleft+zright),2.)
        xerr = old_div((zright-zleft),2.)
        N = Nofzs[np.logical_and(zrange>zleft,zrange<=zright)].sum()
        # print((zleft,zright,N))
        if k==0:
            lab = labres
        else:
            lab = None
            
        
        currentAxis.add_patch(Rectangle((zcent - xerr+pad, 0), 2*xerr-old_div(pad,2.), N, facecolor=col,label=lab))#,alpha=0.5))
        Ndict[labres].append(N)
        k+=1
    
    

pl.legend(labsize=12,loc='upper right')
pl._ax.set_ylim(1,5.e4) 
pl._ax.set_xlim(0.,3.)
pl.done(out_dir+"FigCountsBoth.pdf")
sys.exit()



pl = Plotter(xlabel="$z$",ylabel="Ratio",ftsize=20,figsize=(6,2),yscale='log',thk=2,labsize=16)

colList = ['C0','C1','C2','C3','C4']
Nref = Ndict['S4 3.0\'']
#Nref = Ndict['S4 2.0\'']
expList = ['S4-1.0-paper','S4-1.5-paper','S4-2.0-paper','S4-2.5-paper']
labList = ['S4 1.0\'','S4 1.5\'','S4 2.0\'','S4 2.5\'']

for expName,col,labres in zip(expList,colList,labList):


    currentAxis = plt.gca()

    k = 0
    for zleft,zright in zip(zbins[:-1],zbins[1:]):
        zcent = old_div((zleft+zright),2.)
        xerr = old_div((zright-zleft),2.)
        N = old_div(Ndict[labres][k],Nref[k])
        print((zleft,zright,N))
        if k==0:
            lab = labres
        else:
            lab = None
            
        
        currentAxis.add_patch(Rectangle((zcent - xerr+pad, 1+pad), 2*xerr-old_div(pad,2.), N, facecolor=col,label=lab))#,alpha=0.5))
        k+=1
    
    

#pl.legendOn(labsize=9,loc='upper right')
pl._ax.axhline(y=1.,color=colList[-1],ls="--")
pl._ax.set_ylim(0.9,200) 
pl._ax.set_xlim(0.,3.)
pl.done(out_dir+"FigCountsB.pdf")
