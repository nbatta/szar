import matplotlib
matplotlib.use('Agg')
import numpy as np
from szlib.szcounts import ClusterCosmology,SZ_Cluster_Model,Halo_MF,SampleVariance
from orphics.tools.io import Plotter,dictFromSection,listFromConfig
from ConfigParser import SafeConfigParser 

clusterParams = 'LACluster' # from ini file
cosmologyName = 'LACosmology' # from ini file
iniFile = "input/params.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)

lmax = 3000
cosmoDict = dictFromSection(Config,cosmologyName)
constDict = dictFromSection(Config,'constants')
clusterDict = dictFromSection(Config,clusterParams)
cc = ClusterCosmology(cosmoDict,constDict,lmax,pickling=True)

mrange = np.arange(14.,15.,0.1)
zrange = np.arange(0.05,1.0,0.1)

sv = SampleVariance(cc,mrange,zrange)
hb = sv.haloBias()
print hb.shape

sv.sample_variance_overNsquared(fsky=0.5,lmax=2000)
