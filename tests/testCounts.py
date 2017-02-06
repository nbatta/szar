import matplotlib
matplotlib.use('Agg')
import camb
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import sys, os, time
from szlib.szcounts import ClusterCosmology,SZ_Cluster_Model,Halo_MF
from orphics.tools.io import Plotter,dictFromSection,listFromConfig
from ConfigParser import SafeConfigParser 

zz = 0.5
MM = 5.e14
clusterParams = 'LACluster' # from ini file
cosmologyName = 'LACosmology' # from ini file
experimentName = "LAExp"
fileFunc = None
#fileFunc = lambda M,z:"data/"+experimentName+"_m"+str(M)+"z"+str(z)+".txt"

iniFile = "input/params.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)


beam = listFromConfig(Config,experimentName,'beams')
noise = listFromConfig(Config,experimentName,'noises')
freq = listFromConfig(Config,experimentName,'freqs')
lmax = int(Config.getfloat(experimentName,'lmax'))
lknee = Config.getfloat(experimentName,'lknee')
alpha = Config.getfloat(experimentName,'alpha')
fsky = Config.getfloat(experimentName,'fsky')



# accuracy params
dell=10
pmaxN=5
numps=1000
tmaxN=5
numts=1000

# dell=1
# pmaxN=25
# numps=10000
# tmaxN=25
# numts=10000





cosmoDict = dictFromSection(Config,cosmologyName)
#cosmoDict = dictFromSection(Config,'WMAP9')
constDict = dictFromSection(Config,'constants')
clusterDict = dictFromSection(Config,clusterParams)
cc = ClusterCosmology(cosmoDict,constDict,lmax)

# make an SZ profile example


SZProfExample = SZ_Cluster_Model(clusterCosmology=cc,clusterDict=clusterDict,rms_noises = noise,fwhms=beam,freqs=freq,lmax=lmax,lknee=lknee,alpha=alpha,dell=dell,pmaxN=pmaxN,numps=numps)


print "quickvar " , np.sqrt(SZProfExample.quickVar(MM,zz,tmaxN=tmaxN,numts=numts))
#print "filtvar " , np.sqrt(SZProfExample.filter_variance(MM,zz))





print "y_m",SZProfExample.Y_M(MM,zz)


R500 = cc.rdel_c(MM,zz,500.).flatten()[0]
print R500
DAz = cc.results.angular_diameter_distance(zz) * (cc.H0/100.) 
thetc = R500/DAz
print "thetc = ", thetc



# HMF

zbin_temp = np.arange(0.1,2.0,0.1)
zbin = np.insert(zbin_temp,0,0.0)
#print zbin


start3 = time.time()

HMF = Halo_MF(clusterCosmology=cc)
dvdz = HMF.dVdz(zbin)
dndm = HMF.N_of_z_SZ(zbin,beam,noise,freq,clusterDict,lknee,alpha,fileFunc)

print "Time for N of z " , time.time() - start3


pl = Plotter()
pl.add(zbin[1:], dndm * dvdz[1:])
pl.done("output/dndm.png")

print "Total number of clusters ", np.trapz(dndm * dvdz[1:],zbin[1:],np.diff(zbin[1:]))*4.*np.pi*fsky

#np.savetxt('output/dndm_dVdz_1muK_3_0arc.txt',np.transpose([zbin[1:],dndm,dvdz[1:]]))

