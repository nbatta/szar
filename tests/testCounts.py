import matplotlib
matplotlib.use('Agg')
import camb
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import sys, os, time
from szlib.szcounts import ClusterCosmology,SZ_Cluster_Model,Halo_MF,dictFromSection

from orphics.tools.output import Plotter
from ConfigParser import SafeConfigParser 

beam = [1.5]
noise = [1.0]
freq = [150.]
lmax = 10000 
zz = 0.5
MM= 5e14
clusterParams = 'LACluster' # from ini file


# accuracy params
#dell=10
#pmaxN=5
#numps=1000
#tmaxN=5
#numts=1000

dell=1
pmaxN=25
numps=5000
tmaxN=25
numts=5000




iniFile = "input/cosmology.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)

cosmoDict = dictFromSection(Config,'LACosmology')
#cosmoDict = dictFromSection(Config,'WMAP9')
constDict = dictFromSection(Config,'constants')
clusterDict = dictFromSection(Config,clusterParams)
cc = ClusterCosmology(cosmoDict,constDict,lmax)

spec_file = Config.get('general','Clfile')

# make an SZ profile example


SZProfExample = SZ_Cluster_Model(clusterCosmology=cc,spec_file=spec_file,clusterDict=clusterDict,rms_noises = noise,fwhms=beam,freqs=freq,lmax=lmax,M=MM,z=zz,dell=dell,pmaxN=pmaxN,numps=numps)


print "quickvar " , np.sqrt(SZProfExample.quickVar(MM,zz,tmaxN=tmaxN,numts=numts))
print "filtvar " , np.sqrt(SZProfExample.filter_variance(MM,zz))
print "Y_M ",SZProfExample.Y_M(MM,zz)









sys.exit()
print "y_m",SZProfExample.Y_M(MM,zz)
#SZProfExample.plot_noise()

results = cc.results



R500 = SZProfExample.R500
print R500
thetc = R500/DA_z
#dell = 1
#ells = np.arange(2,20000,dell)
arc = 5.9
#thetc = np.deg2rad(arc / 60.)
thetc2 = np.deg2rad(arc/2. / 60.)
print "thetc = ", thetc



DA_z = results.angular_diameter_distance(zz) * (cc.H0/100.)

start2 = time.time()
print np.sqrt(SZProfExample.filter_variance(DA_z,newMethod=newMethod))
print "Time for var " , time.time() - start2

pl = Plotter()
pl._ax.set_xlim(0,10)
pl.add(np.rad2deg(tht)*60.,filt/np.max(filt),color='black')
pl.add(np.rad2deg(tht)*60.,SZProfExample.y2D_norm(tht/thetc),color="black",ls='--')
pl.add([0,10],[0,0],color="black",ls='--')
pl.done("output/filter.png")



# HMF

zbin_temp = np.arange(0.1,0.8,0.2)
zbin = np.insert(zbin_temp,0,0.0)
print zbin


start3 = time.time()

HMF = Halo_MF(clusterCosmology=cc)
dvdz = HMF.dVdz(zbin)
dndm = HMF.N_of_z_SZ(zbin,beam,noise,spec_file,clusterDict,fileFunc = lambda beam,noise,Mexp,z:"data/m"+str(Mexp)+"z"+str(z)+"b"+str(beam)+"n"+str(noise)+".txt")

print "Time for N of z " , time.time() - start3


pl = Plotter()
pl.add(zbin[1:], dndm * dvdz[1:])
pl.done("output/dndm.png")

#np.savetxt('output/dndm_dVdz_1muK_3_0arc.txt',np.transpose([zbin[1:],dndm,dvdz[1:]]))

