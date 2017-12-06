import szar.likelihood as lk
import matplotlib.pyplot as plt
from ConfigParser import SafeConfigParser
from orphics.tools.io import dictFromSection

import time

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

#fparams = {}
#for (key, val) in Config.items('params'):
#    if ',' in val:
#        param, step = val.split(',')
#        fparams[key] = float(param)
#    else:
#        fparams[key] = float(val)

nemoOutputDir = '/Users/nab/Desktop/Projects/ACTPol_Cluster_Like/ACTdata/'
pardict = nemoOutputDir + 'equD56.par'

CL = lk.clusterLike(iniFile,expName,gridName,pardict,nemoOutputDir)

