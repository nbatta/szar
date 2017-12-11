import numpy as np
from szar.counts import ClusterCosmology,Halo_MF
from szar.szproperties import SZ_Cluster_Model
from configparser import SafeConfigParser
from orphics.tools.io import dictFromSection,listFromConfig
import cPickle as pickle


iniFile = 'input/pipeline.ini'
expName = 'CCATP-propv2'
gridName = 'grid-owl2'
cal = 'owl2'

Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)

fparams = {}
for (key, val) in Config.items('params'):
    if ',' in val:
        param, step = val.split(',')
        fparams[key] = float(param)
    else:
        fparams[key] = float(val)
        
bigDataDir = Config.get('general','bigDataDirectory')
clttfile = Config.get('general','clttfile')
constDict = dictFromSection(Config,'constants')
clusterDict = dictFromSection(Config,'cluster_params')
beam = listFromConfig(Config,expName,'beams')
noise = listFromConfig(Config,expName,'noises')
freq = listFromConfig(Config,expName,'freqs')
lknee = listFromConfig(Config,expName,'lknee')[0]
alpha = listFromConfig(Config,expName,'alpha')[0]

calFile = bigDataDir+"lensgrid_grid-"+cal+"_"+cal+".pkl"

version = Config.get('general','version')

mexp_edges, z_edges, lndM = pickle.load(open(calFile,"rb"))

mgrid,zgrid,siggrid = pickle.load(open(bigDataDir+"szgrid_"+expName+"_"+gridName+ "_v" + version+".pkl",'rb'))

assert np.all(mgrid==mexp_edges)
assert np.all(z_edges==zgrid)

cc = ClusterCosmology(fparams,constDict,clTTFixFile=clttfile)
HMF = Halo_MF(cc,mgrid,zgrid)

qs = listFromConfig(Config,'general','qbins')
qspacing = Config.get('general','qbins_spacing')
if qspacing=="log":
    qbin_edges = np.logspace(np.log10(qs[0]),np.log10(qs[1]),int(qs[2])+1)
elif qspacing=="linear":
    qbin_edges = np.linspace(qs[0],qs[1],int(qs[2])+1)
else:
    raise ValueError

SZProf = SZ_Cluster_Model(cc,clusterDict,rms_noises = noise,fwhms=beam,freqs=freq,lknee=lknee,alpha=alpha)
dN_dmqz_corr = HMF.N_of_mqz_SZ_corr(lndM,qbin_edges,SZProf)
dN_dmqz = HMF.N_of_mqz_SZ(lndM,qbin_edges,SZProf)

