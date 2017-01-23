from szlib.szcounts import ClusterCosmology,SZ_Cluster_Model,dictFromSection
import sys
import os
from ConfigParser import SafeConfigParser
import numpy as np

beam = float(sys.argv[1])
noise = float(sys.argv[2])
Mexp = float(sys.argv[3])
M = 10.**Mexp
z = float(sys.argv[4])

iniFile = "input/cosmology.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)

cosmoDict = dictFromSection(Config,'WMAP9')
constDict = dictFromSection(Config,'constants')
cc = ClusterCosmology(cosmoDict,constDict)


# make an SZ profile example
SZProf = SZ_Cluster_Model(clusterCosmology=cc,rms_noise = noise,fwhm=beam,M=M,z=z)
DA_z = cc.results.angular_diameter_distance(z) * (cc.H0/100.)
sigN = np.sqrt(SZProf.filter_variance(DA_z))
            
np.savetxt("data/m"+str(Mexp)+"z"+str(z)+"b"+str(beam)+"n"+str(noise)+".txt",np.array([Mexp,z,beam,noise,sigN]))
