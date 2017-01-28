import matplotlib
matplotlib.use('Agg')
import camb
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import sys, os, time
from szlib.szcounts import ClusterCosmology,SZ_Cluster_Model,Halo_MF,dictFromSection,listFromConfig

from orphics.tools.output import Plotter
from ConfigParser import SafeConfigParser 

zz = 0.5
MM = 5.e14
clusterParams = 'LACluster' # from ini file
cosmologyName = 'LACosmology' # from ini file

#fileFunc = None
fileFunc = lambda M,z:"data/"+experimentName+"_m"+str(M)+"z"+str(z)+".txt"
experimentName = ["S45m","S46m","S47m"]
#experimentName = ["SO5m","SO6m","SO7m","SO5m_No270","SO6m_No270","SO7m_No270"]

for ii in xrange(len(experimentName)):

    iniFile = "input/params.ini"
    Config = SafeConfigParser()
    Config.optionxform=str
    Config.read(iniFile)


    beam = listFromConfig(Config,experimentName[ii],'beams')
    noise = listFromConfig(Config,experimentName[ii],'noises')
    freq = listFromConfig(Config,experimentName[ii],'freqs')
    lmax = int(Config.getfloat(experimentName[ii],'lmax'))
    lknee = Config.getfloat(experimentName[ii],'lknee')
    alpha = Config.getfloat(experimentName[ii],'alpha')
    fsky = Config.getfloat(experimentName[ii],'fsky')

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

# HMF

    zbin_temp = np.arange(0.05,2.0,0.05)
    zbin = np.insert(zbin_temp,0,0.0)

    start3 = time.time()

    HMF = Halo_MF(clusterCosmology=cc)
    dvdz = HMF.dVdz(zbin)
    #dndm = HMF.N_of_z_SZ(zbin,beam,noise,freq,clusterDict,lknee,alpha,fileFunc)
    dNdmdz,dm = HMF.N_of_mz_SZ(zbin,beam,noise,freq,clusterDict,lknee,alpha,fileFunc)

    print "Time for N of z " , time.time() - start3

    #pl = Plotter()
    #pl.add(zbin[1:], dndm * dvdz[1:])
    #pl.done("output/dndm"+experimentName[ii]+".png")

    #print "Total number of clusters ", np.trapz(dndm * dvdz[1:],zbin[1:],np.diff(zbin[1:]))*4.*np.pi*fsky

#    np.savetxt('output/dN_dz_'+experimentName[ii]+'.txt',np.transpose([zbin[1:],dndm,dvdz[1:]])) 
    np.savetxt('output/dN_dmdz_'+experimentName[ii]+'.txt',np.transpose([zbin[1:],dm,dNdmdz,dvdz[1:]])) 
#np.savetxt('output/dndm_dVdz_1muK_3_0arc.txt',np.transpose([zbin[1:],dndm,dvdz[1:]]))

