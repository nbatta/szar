import numpy as np
import orphics.tools.io as io
import sys, time
from szar.counts import ClusterCosmology,Halo_MF
from configparser import SafeConfigParser

paramName = sys.argv[1]
key = paramName
print(("Calculating derivative for ", key))

iniFile = "input/params.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)

cosmologyName = 'LACosmology' # from ini file
clusterParams = 'LACluster' # from ini file   
cosmoListDict = io.dictOfListsFromSection(Config,cosmologyName)
cosmoDict = io.dictFromSection(Config,cosmologyName)
constDict = io.dictFromSection(Config,'constants')
clusterDict = io.dictFromSection(Config,clusterParams)

experimentName = "AdvAct"

saveId = experimentName + cosmologyName
upDict = cosmoDict.copy()
dnDict = cosmoDict.copy()

try:
    upDict[key] = cosmoListDict[key][0] + cosmoListDict[key][1]/2.
    dnDict[key] = cosmoListDict[key][0] - cosmoListDict[key][1]/2.
except:
    print(("No step size specified for ", key))
    sys.exit()    

print((upDict[key]))
print((dnDict[key]))

beam = io.listFromConfig(Config,experimentName,'beams')
noise = io.listFromConfig(Config,experimentName,'noises')
freq = io.listFromConfig(Config,experimentName,'freqs')
lmax = int(Config.getfloat(experimentName,'lmax'))
lknee = Config.getfloat(experimentName,'lknee')
alpha = Config.getfloat(experimentName,'alpha')
fsky = Config.getfloat(experimentName,'fsky')

mass_err_file = Config.get(experimentName,'mass_err')
mass_err = np.loadtxt(mass_err_file)


ccUp = ClusterCosmology(upDict,constDict,lmax)
ccDn = ClusterCosmology(dnDict,constDict,lmax)

#mbin = np.arange(12.5,15.5,0.05)+0.05
#zbin_temp = np.arange(0.05,2.05,0.05)

zbin_temp = np.arange(0.05,2.0,0.05)
zbin = np.insert(zbin_temp,0,0.0)
qbin = np.arange(np.log(6),np.log(500),0.08)
mbin = np.arange(13.5, 15.71, 0.1)

start3 = time.time()

HMFup = Halo_MF(clusterCosmology=ccUp)
dN_dmqz_up = HMFup.N_of_mqz_SZ(mass_err,zbin,mbin,np.exp(qbin),beam,noise,freq,clusterDict,lknee,alpha)
print((dN_dmqz_up.sum()))
HMFdn = Halo_MF(clusterCosmology=ccDn)
dN_dmqz_dn = HMFdn.N_of_mqz_SZ(mass_err,zbin,mbin,np.exp(qbin),beam,noise,freq,clusterDict,lknee,alpha)

print(("Time for N of z " , time.time() - start3))

np.save("data/dN_dzmq"+saveId+"_"+key+"_up",dN_dmqz_up)
np.save("data/dN_dzmq"+saveId+"_"+key+"_dn",dN_dmqz_dn)

               
