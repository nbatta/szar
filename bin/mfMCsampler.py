import numpy as np
from szar.counts import ClusterCosmology,Halo_MF
from orphics.tools.io import dictFromSection, listFromConfig
from ConfigParser import SafeConfigParser
import cPickle as pickle
import matplotlib.pyplot as plt
import emcee


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

bla = HMF.N_of_Mz(HMF.M200,200)
print HMF.M[100]
print HMF.zarr[18]

print bla[100][18]

blah = HMF.sample_mf(200)#(1.8,4e14,bla)

print blah(HMF.zarr[18],HMF.M[100])

plt.imshow(np.log10(bla))
plt.colorbar()
plt.savefig('default.png', bbox_inches='tight',format='png')

