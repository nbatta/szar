"""                                                                                                 
                                                                                                    
Calculates cluster selction function 
                                                                                                    
python bin/get_selection_func.py <paramList> <expName> <calName> <calFile>                                  
                                                                                                    
<paramList> is comma separated param list, no spaces, case-sensitive.                               
                                                                                                    
If <paramList> is "allParams", calculates derivatives for all                                       
params with step sizes in [params] section of ini file.                                             
                                                                                                    
<expName> is name of section in input/pipelineMakeDerivs.py                                         
that specifies an experiment.                                                                       
                                                                                                    
<calName> name of calibration that will be used in the saved files                                  
                                                                                                    
<calFile> is the name of a pickle file containing the mass                                          
calibration error over mass.                                                                        
                                                                                                    
"""

from szar.counts import ClusterCosmology,Halo_MF
import numpy as np
from szar.szproperties import SZ_Cluster_Model
import sys
from configparser import SafeConfigParser
import pickle as pickle
from orphics.io import dict_from_section, list_from_config

expName = sys.argv[1]
gridName = sys.argv[2]
calName = sys.argv[3]

iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)
bigDataDir = Config.get('general','bigDataDirectory')

calFile = bigDataDir+"lensgrid_grid-"+calName+"_"+calName+".pkl"

version = Config.get('general','version')
# load the mass calibration grid                                                                
mexp_edges, z_edges, lndM = pickle.load(open(calFile,"rb"),encoding='latin1')

saveId = expName + "_" + gridName + "_" + calName + "_v" + version

constDict = dict_from_section(Config,'constants')
clttfile = Config.get('general','clttfile')
clusterDict = dict_from_section(Config,'cluster_params')
beam = list_from_config(Config,expName,'beams')
noise = list_from_config(Config,expName,'noises')
freq = list_from_config(Config,expName,'freqs')
lknee = list_from_config(Config,expName,'lknee')[0]
alpha = list_from_config(Config,expName,'alpha')[0]
fsky = Config.getfloat(expName,'fsky')
try:
    v3mode = Config.getint(expName,'V3mode')
except:
    v3mode = -1

fparams = {}
for (key, val) in Config.items('params'):
    if ',' in val:
        param, step = val.split(',')
        fparams[key] = float(param)
    else:
        fparams[key] = float(val)

cc = ClusterCosmology(fparams,constDict,clTTFixFile=clttfile)
HMF = Halo_MF(cc,mexp_edges,z_edges)
SZCluster = SZ_Cluster_Model(cc,clusterDict,rms_noises = noise,fwhms=beam,freqs=freq,lknee=lknee,alpha=alpha,v3mode=v3mode,fsky=fsky)

HMF.updateSigN(SZCluster)
HMF.updatePfunc(SZCluster)

np.save(bigDataDir+"Sel_func"+saveId,HMF.Pfunc)
