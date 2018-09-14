from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import camb
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import sys, os, time
from szar.counts import ClusterCosmology,Halo_MF,getTotN
from szar.szproperties import SZ_Cluster_Model
from orphics.io import Plotter,dict_from_section,list_from_config
from configparser import SafeConfigParser 
from orphics.maps import interpolate_grid
import pickle as pickle

zz = 0.5
MM = 5.e14
clusterParams = 'LACluster' # from ini file
cosmologyName = 'LACosmology' # from ini file
experimentName = "v3test"

iniFile = "input/params.ini"
#iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)


beam = list_from_config(Config,experimentName,'beams')
noise = list_from_config(Config,experimentName,'noises')
freq = list_from_config(Config,experimentName,'freqs')
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





cosmoDict = dict_from_section(Config,cosmologyName)
#cosmoDict = dict_from_section(Config,'WMAP9')
constDict = dict_from_section(Config,'constants')
clusterDict = dict_from_section(Config,clusterParams)
cc = ClusterCosmology(cosmoDict,constDict,pickling=True,clTTFixFile = "data/cltt_lensed_Feb18.txt")

# make an SZ profile example


SZProfExample = SZ_Cluster_Model(clusterCosmology=cc,clusterDict=clusterDict,rms_noises = noise,fwhms=beam,freqs=freq,lmax=lmax,lknee=lknee,alpha=alpha,dell=dell,pmaxN=pmaxN,numps=numps,v3mode=2,fsky=0.4)


#MM = 10**np.linspace(13.,14.,5)
#print SZProfExample.quickVar(MM,zz,tmaxN=tmaxN,numts=numts)

#sys.exit()


print(("quickvar " , np.sqrt(SZProfExample.quickVar(MM,zz,tmaxN=tmaxN,numts=numts))))
#print "filtvar " , np.sqrt(SZProfExample.filter_variance(MM,zz))






print(("y_m",SZProfExample.Y_M(MM,zz)))


R500 = cc.rdel_c(MM,zz,500.).flatten()[0]
print(R500)
print((cc.rhoc(0)))
print((cc.rhoc(zz)))
#sys.exit()
DAz = cc.results.angular_diameter_distance(zz) * (cc.H0/100.) 
thetc = R500/DAz
print(("thetc = ", thetc))



# HMF

zbin = np.arange(0.,3.0,0.1)
#zbin = np.insert(zbin_temp,0,0.0)
#print zbin

Mexp = np.arange(13.5,15.71,0.01)

start3 = time.time()
SZProf = SZ_Cluster_Model(cc,clusterDict,rms_noises = noise,fwhms=beam,freqs=freq,lknee=lknee,alpha=alpha)

HMF = Halo_MF(cc,Mexp,zbin)
dvdz = HMF.dVdz#(zbin)
dndm = HMF.N_of_z_SZ(fsky,SZProf)

sys.exit()
print(("Time for N of z " , time.time() - start3))


# pl = Plotter()
# pl.add(zbin[1:], dndm * dvdz[1:])
# pl.done("output/dndm.png")

print(("Total number of clusters ", np.trapz(dndm ,zbin[:],np.diff(zbin[:]))*fsky))

#np.savetxt('output/dndm_dVdz_1muK_3_0arc.txt',np.transpose([zbin[1:],dndm,dvdz[1:]]))

mfile = "data/S4-7mCMB_all.pkl"
minrange, zinrange, lndM = pickle.load(open(mfile,'rb'))

outmerr = interpolate_grid(lndM,minrange,zinrange,Mexp,zbin,regular=False,kind="cubic",bounds_error=False,fill_value=np.inf)


q_arr = np.logspace(np.log10(6.),np.log10(500.),64)

dnqmz = HMF.N_of_mqz_SZ(outmerr,q_arr,SZProf)

N,Nofz = getTotN(dnqmz,Mexp,zbin,q_arr,returnNz=True)

print((N*fsky))
