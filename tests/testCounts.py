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

beam = 1.5
noise = 1.0
moreElls = True # uses the ell up to 60000 code, so no lmax, and uses cosmology from spec_file. \
           # Otherwise uses cosmology from internal CAMB call pickled every day.
lmax = 8000 # doesn't matter if moreElls is True
newMethod = True # filterVariance returns value from the lambda function implementation
clusterParams = 'AlonsoCluster' # from ini file

iniFile = "input/cosmology.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)

cosmoDict = dictFromSection(Config,'WMAP9')
constDict = dictFromSection(Config,'constants')
clusterDict = dictFromSection(Config,clusterParams)
cc = ClusterCosmology(cosmoDict,constDict,lmax)

spec_file = Config.get('general','Clfile')

# make an SZ profile example
SZProfExample = SZ_Cluster_Model(clusterCosmology=cc,spec_file=spec_file,clusterDict=clusterDict,rms_noise = noise,moreElls=moreElls,fwhm=beam,lmax=lmax,M=5e14,z=0.5 )
zz = 0.5
MM= 5e14
print "y_m",SZProfExample.Y_M(MM,zz)
SZProfExample.plot_noise()

results = cc.results

DA_z = results.angular_diameter_distance(zz) * (cc.H0/100.)


R500 = SZProfExample.R500
print R500
thetc = R500/DA_z
#dell = 1
#ells = np.arange(2,20000,dell)
arc = 5.9
#thetc = np.deg2rad(arc / 60.)
thetc2 = np.deg2rad(arc/2. / 60.)
print "thetc = ", thetc

el, nltemp, cl = SZProfExample.tot_noise_spec(spec_file)

dell = 100
ells = np.arange(2,60000,dell)
nl = np.interp(ells,el,nltemp)
start = time.time()


dtht = 0.0000025
thta = np.arange(dtht,5.*thetc,dtht)

y2dt2,y2D_use= SZProfExample.y2D_tilde_norm(ells,thetc,thta)
print "Time for norm ", time.time() - start


pl = Plotter(scaleX='log', scaleY='log')
pl.add(ells,nl,ls='--')
pl.add(ells,y2dt2)
pl.add(ells,y2dt2/nl)
#plt.loglog(ells,y2dt2_2)
pl.done("output/ynorm.png")


DA_z = results.angular_diameter_distance(zz) * (cc.H0/100.)

start2 = time.time()
print np.sqrt(SZProfExample.filter_variance(DA_z))
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

