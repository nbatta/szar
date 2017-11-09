import numpy as np
from szar.counts import ClusterCosmology,Halo_MF
from orphics.tools.io import dictFromSection, listFromConfig
from ConfigParser import SafeConfigParser
import cPickle as pickle

#initial setup

iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)
bigDataDir = Config.get('general','bigDataDirectory')
clttfile = Config.get('general','clttfile')
constDict = dictFromSection(Config,'constants')
version = Config.get('general','version')
expName = "S4-1.0-CDT"
gridName = "grid-owl2"

fparams = {}
for (key, val) in Config.items('params'):
    if ',' in val:
        param, step = val.split(',')
        fparams[key] = float(param)
    else:
        fparams[key] = float(val)

mgrid,zgrid,siggrid = pickle.load(open(bigDataDir+"szgrid_"+expName+"_"+gridName+ "_v" + version+".pkl",'rb'))

cc = ClusterCosmology(fparams,constDict,clTTFixFile=clttfile)
HMF = Halo_MF(cc,mgrid,zgrid)

blah = HMF.sample_mf(200)

print blah(1e15,0.8)
