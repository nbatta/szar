import camb
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import sys, os, time
from szlib.szcounts import ClusterCosmology,SZ_Cluster_Model,Halo_MF,dictFromSection,listFromConfig

from orphics.tools.output import Plotter
from ConfigParser import SafeConfigParser 


saveId = "AdvAct"

iniFile = "input/params.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)


beam = listFromConfig(Config,saveId,'beams')
noise = listFromConfig(Config,saveId,'noises')
freq = listFromConfig(Config,saveId,'freqs')
lmax = int(Config.getfloat(saveId,'lmax'))

clusterParams = 'LACluster' # from ini file
cosmologyName = 'LACosmology' # from ini file

# accuracy params
dell=10
pmaxN=5
numps=1000
tmaxN=5
numts=1000





cosmoDict = dictFromSection(Config,cosmologyName)
constDict = dictFromSection(Config,'constants')
clusterDict = dictFromSection(Config,clusterParams)
cc = ClusterCosmology(cosmoDict,constDict,lmax)

# make an SZ profile example


SZProfExample = SZ_Cluster_Model(clusterCosmology=cc,clusterDict=clusterDict,rms_noises = noise,fwhms=beam,freqs=freq,lmax=lmax,dell=dell,pmaxN=pmaxN,numps=numps)



z = float(sys.argv[1])


Mexps = np.arange(14.0,15.5,0.05)

for Mexp in Mexps:
    M = 10.**Mexp
    sigN = np.sqrt(SZProfExample.quickVar(M,z,tmaxN=tmaxN,numts=numts))

    np.savetxt("data/"+saveId+"_m"+str(Mexp)+"z"+str(z)+".txt",np.array([Mexp,z,sigN]))
