import matplotlib
matplotlib.use('Agg')
import sys
import numpy as np
from szlib.szcounts import ClusterCosmology,Halo_MF,sampleVarianceOverNsquareOverBsquare,haloBias,getTotN
#from szlib.szproperties import SZ_Cluster_Model
from orphics.tools.io import Plotter,dictFromSection,listFromConfig
from ConfigParser import SafeConfigParser 

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

constDict = dictFromSection(Config,'constants')
clusterDict = dictFromSection(Config,'cluster_params')
cc = ClusterCosmology(fparams,constDict,skipCls=True)

fsky = Config.getfloat(expName,'fsky')

saveId = expName + "_" + gridName + "_v" + version

ms = listFromConfig(Config,gridName,'mexprange')
mrange = np.arange(ms[0],ms[1]+ms[2],ms[2])
zs = listFromConfig(Config,gridName,'zrange')
zrange = np.arange(zs[0],zs[1]+zs[2],zs[2])


hmf = Halo_MF(cc,mrange,zrange)
zcents, hb = haloBias(mrange,zrange,cc.rhoc0om,hmf.kh,hmf.pk)
powers = sampleVarianceOverNsquareOverBsquare(cc,hmf.kh,hmf.pk,zrange,fsky,lmax=lmax)

sovernsquarebsquare = np.outer(powers,np.ones([len(mrange)-1])).transpose()

sovernsquare = hb*hb*sovernsquarebsquare


np.savetxt(bigDataDir+"sampleVarGrid_"+saveId+".txt",sovernsquare)




