from __future__ import print_function
from __future__ import division
from past.utils import old_div
import matplotlib
matplotlib.use('Agg')
import camb
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import sys, os, time
from szar.counts import ClusterCosmology,Halo_MF,getTotN
from szar.szproperties import SZ_Cluster_Model
from orphics.tools.io import Plotter,dictFromSection,listFromConfig
from configparser import SafeConfigParser 
from orphics.analysis.flatMaps import interpolateGrid
import pickle as pickle

gridName = "grid-owl1-planck"
calName = "owl1-planck" 
zz = 0.5
MM = 5.e14
clusterParams = 'cluster_params' # from ini file
cosmologyName = 'params' # from ini file
experimentName = "PlanckTest"
expName = experimentName
#iniFile = "input/params.ini"
iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)


beam = listFromConfig(Config,experimentName,'beams')
noise = listFromConfig(Config,experimentName,'noises')
freq = listFromConfig(Config,experimentName,'freqs')
lmax = int(Config.getfloat(experimentName,'lmax'))
lknee = float(Config.get(experimentName,'lknee').split(',')[0])
alpha = float(Config.get(experimentName,'alpha').split(',')[0])
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
cc = ClusterCosmology(cosmoDict,constDict,pickling=True,clTTFixFile = "data/cltt_lensed_Feb18.txt")

# make an SZ profile example


SZProfExample = SZ_Cluster_Model(clusterCosmology=cc,clusterDict=clusterDict,rms_noises = noise,fwhms=beam,freqs=freq,lmax=lmax,lknee=lknee,alpha=alpha,dell=dell,pmaxN=pmaxN,numps=numps,qmin=6)

version = Config.get('general','version')
bigDataDir = Config.get('general','bigDataDirectory')

calFile = bigDataDir+"lensgrid_"+gridName+"_"+calName+".pkl"

mgrid,zgrid,siggrid = pickle.load(open(bigDataDir+"szgrid_"+expName+"_"+gridName+ "_v" + version+".pkl",'rb'))

Mexp_edges, z_edges, lndM = pickle.load(open(calFile,"rb"))

HMF = Halo_MF(cc,Mexp_edges,z_edges)
HMF.sigN = siggrid.copy()
#MM = 10**np.linspace(13.,14.,5)
#print SZProfExample.quickVar(MM,zz,tmaxN=tmaxN,numts=numts)

#sys.exit()

#print z_edges
#print HMF.N_of_z()

Nzs =  HMF.N_of_z_SZ(fsky,SZProfExample)*np.diff(z_edges)
zcents = old_div((z_edges[1:]+z_edges[:-1]),2.)
pl = Plotter()
pl.add(zcents,Nzs)
pl.done("nz.png")


print((HMF.Mass_err(fsky,lndM*24.0,SZProfExample)))

#print "quickvar " , np.sqrt(SZProfExample.quickVar(MM,zz,tmaxN=tmaxN,numts=numts))
#print "filtvar " , np.sqrt(SZProfExample.filter_variance(MM,zz))






#print "y_m",SZProfExample.Y_M(MM,zz)


#R500 = cc.rdel_c(MM,zz,500.).flatten()[0]
#print R500
#print cc.rhoc(0)
#print cc.rhoc(zz)
sys.exit()
#DAz = cc.results.angular_diameter_distance(zz) * (cc.H0/100.) 
#thetc = R500/DAz
#print "thetc = ", thetc



# HMF

#zbin = np.arange(0.,3.0,0.1)
#zbin = np.insert(zbin_temp,0,0.0)
#print zbin

Mexp = np.arange(13.5,15.71,0.01)

start3 = time.time()
SZProf = SZ_Cluster_Model(cc,clusterDict,rms_noises = noise,fwhms=beam,freqs=freq,lknee=lknee,alpha=alpha)

HMF = Halo_MF(cc,Mexp,zbin)
dvdz = HMF.dVdz#(zbin)
dndm = HMF.N_of_z_SZ(SZProf)

sys.exit()
print(("Time for N of z " , time.time() - start3))


# pl = Plotter()
# pl.add(zbin[1:], dndm * dvdz[1:])
# pl.done("output/dndm.png")

print(("Total number of clusters ", np.trapz(dndm ,zbin[:],np.diff(zbin[:]))*fsky))

#np.savetxt('output/dndm_dVdz_1muK_3_0arc.txt',np.transpose([zbin[1:],dndm,dvdz[1:]]))

mfile = "data/S4-7mCMB_all.pkl"
minrange, zinrange, lndM = pickle.load(open(mfile,'rb'))

outmerr = interpolateGrid(lndM,minrange,zinrange,Mexp,zbin,regular=False,kind="cubic",bounds_error=False,fill_value=np.inf)


q_arr = np.logspace(np.log10(6.),np.log10(500.),64)

dnqmz = HMF.N_of_mqz_SZ(outmerr,q_arr,SZProf)

N,Nofz = getTotN(dnqmz,Mexp,zbin,q_arr,returnNz=True)

print((N*fsky))
