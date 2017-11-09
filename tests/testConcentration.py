import matplotlib
matplotlib.use('Agg')
from szar.counts import ClusterCosmology
import numpy as np
import sys,os
from configparser import SafeConfigParser 
import pickle as pickle
from orphics.tools.io import dictFromSection, listFromConfig, Plotter
import szar._fast as fast

M = 2.e14
z = 0.1

iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)

constDict = dictFromSection(Config,'constants')
fparams = {}   # the 
for (key, val) in Config.items('params'):
    if ',' in val:
        param, step = val.split(',')
        fparams[key] = float(param)
    else:
        fparams[key] = float(val)

clttfile = Config.get('general','clttfile')
#fparams['H0']=100.
cc = ClusterCosmology(fparams,constDict,clTTFixFile=clttfile)


print((cc.Mdel_to_cdel(M,z,500.)))
