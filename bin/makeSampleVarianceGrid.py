import matplotlib
matplotlib.use('Agg')
import sys
import numpy as np
from szar.counts import ClusterCosmology,Halo_MF,sampleVarianceOverNsquareOverBsquare,haloBias,getTotN
#from szar.szproperties import SZ_Cluster_Model
from orphics.io import Plotter,dict_from_section,list_from_config
from configparser import SafeConfigParser 

lmax = 1000

expName = sys.argv[1]
gridName = sys.argv[2]


iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)
bigDataDir = Config.get('general','bigDataDirectory')
version = Config.get('general','version')

fparams = {}   # the 
for (key, val) in Config.items('params'):
    if ',' in val:
        param, step = val.split(',')
        fparams[key] = float(param)
    else:
        fparams[key] = float(val)

constDict = dict_from_section(Config,'constants')
clusterDict = dict_from_section(Config,'cluster_params')
cc = ClusterCosmology(fparams,constDict,skipCls=True)

fsky = Config.getfloat(expName,'fsky')

saveId = expName + "_" + gridName + "_v" + version

ms = list_from_config(Config,gridName,'mexprange')
mrange = np.arange(ms[0],ms[1]+ms[2],ms[2])
zs = list_from_config(Config,gridName,'zrange')
zrange = np.arange(zs[0],zs[1]+zs[2],zs[2])


hmf = Halo_MF(cc,mrange,zrange)
zcents, hb = haloBias(mrange,zrange,cc.rhoc0om,hmf.kh,hmf.pk)
powers = sampleVarianceOverNsquareOverBsquare(cc,hmf.kh,hmf.pk,zrange,fsky,lmax=lmax)

sovernsquarebsquare = np.outer(powers,np.ones([len(mrange)-1])).transpose()

sovernsquare = hb*hb*sovernsquarebsquare


np.savetxt(bigDataDir+"sampleVarGrid_"+saveId+".txt",sovernsquare)




