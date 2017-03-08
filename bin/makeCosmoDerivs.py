import matplotlib
matplotlib.use('Agg')
import numpy as np
import sys, os, time
from szlib.szcounts import ClusterCosmology,SZ_Cluster_Model,Halo_MF,getTotN
from orphics.tools.io import Plotter,dictFromSection,listFromConfig
from ConfigParser import SafeConfigParser 
import cPickle as pickle
from orphics.tools.io import Plotter
from orphics.analysis.flatMaps import interpolateGrid




inParamList = sys.argv[1].split(',')

iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)


paramList = [] # the parameters that can be varied
fparams = {}   # the 
stepSizes = {}
for (key, val) in Config.items('params'):
    if ',' in val:
        param, step = val.split(',')
        paramList.append(key)
        fparams[key] = float(param)
        stepSizes[key] = float(step)
    else:
        fparams[key] = float(val)

cosmologicalParams = Config.get('param-info','cosmological').split(',')
scalingParams = Config.get('param-info','scaling').split(',')

if inParamList[0]=="allParams":
    assert len(inParamList)==1, "I'm confused why you'd specify more params with allParams."

    inParamList = paramList

else:
    for param in inParamList:
        assert param in paramList, param + " not found in ini file with a specified step size."
        assert param in stepSizes.keys(), param + " not found in stepSizes dict. Looks like a bug in the code."

for param in inParamList:
    assert (param in cosmologicalParams) or (param in scalingParams), param + ' not found in list of cosmological or scaling params in ini file param-info section.'



cc = ClusterCosmology(fparams,constDict,clTTFixFile=clttfile)
HMF = Halo_MF(cc,mexprange,zrange)
SZProf = SZ_Cluster_Model(cc,clusterDict,rms_noises = noise,fwhms=beam,freqs=freq,lknee=lknee,alpha=alpha)
dNFid_dmqz = HMF.N_of_mqz_SZ(lndM,qbins,SZProf)


