from szar.counts import ClusterCosmology,Halo_MF,getNmzq
from szar.szproperties import SZ_Cluster_Model
import szar.fisher as sfisher
import numpy as np
import sys
from configparser import SafeConfigParser 
import pickle as pickle
    
expName = sys.argv[1]
gridName = sys.argv[2]
calName = sys.argv[3]

iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)
bigDataDir = Config.get('general','bigDataDirectory')
version = Config.get('general','version')

fparams = {}   # the 
stepSizes = {}
for (key, val) in Config.items('params'):
    if ',' in val:
        param, step = val.split(',')
        fparams[key] = float(param)
        stepSizes[key] = float(step)
    else:
        fparams[key] = float(val)




rayStep = stepSizes['sigR']

calFileUp = sfisher.mass_grid_name_cmb_up(bigDataDir,expName,gridName,calName,version) 
calFileDn = sfisher.mass_grid_name_cmb_dn(bigDataDir,expName,gridName,calName,version)
    
# load the mass calibration grid
mexp_edges, z_edges, lndMUp = pickle.load(open(calFileUp,"rb"))
mexp_edgesDn, z_edgesDn, lndMDn = pickle.load(open(calFileDn,"rb"))
assert np.all(np.isclose(mexp_edges,mexp_edgesDn))
assert np.all(np.isclose(z_edges,z_edgesDn))

mgrid,zgrid,siggrid = pickle.load(open(bigDataDir+"szgrid_"+expName+"_"+gridName+ "_v" + version+".pkl",'rb'))
assert np.all(np.isclose(mgrid,mexp_edges))
assert np.all(np.isclose(z_edges,zgrid))
    
saveId = expName + "_" + gridName + "_" + calName + "_v" + version

from orphics.tools.io import dictFromSection, listFromConfig
constDict = dictFromSection(Config,'constants')
clusterDict = dictFromSection(Config,'cluster_params')
beam = listFromConfig(Config,expName,'beams')
noise = listFromConfig(Config,expName,'noises')
freq = listFromConfig(Config,expName,'freqs')
lknee = listFromConfig(Config,expName,'lknee')[0]
alpha = listFromConfig(Config,expName,'alpha')[0]

clttfile = Config.get('general','clttfile')

# get s/n q-bins
qs = listFromConfig(Config,'general','qbins')
qspacing = Config.get('general','qbins_spacing')
if qspacing=="log":
    qbin_edges = np.logspace(np.log10(qs[0]),np.log10(qs[1]),int(qs[2])+1)
elif qspacing=="linear":
    qbin_edges = np.linspace(qs[0],qs[1],int(qs[2])+1)
else:
    raise ValueError

massMultiplier = Config.getfloat('general','mass_calib_factor')

print("Broadcasting...")
print("Broadcasted.")

cc = ClusterCosmology(fparams,constDict,clTTFixFile=clttfile)
HMF = Halo_MF(cc,mexp_edges,z_edges)
HMF.sigN = siggrid.copy()
SZProf = SZ_Cluster_Model(cc,clusterDict,rms_noises = noise,fwhms=beam,freqs=freq,lknee=lknee,alpha=alpha)
dN_dmqz = HMF.N_of_mqz_SZ(lndMUp*massMultiplier,qbin_edges,SZProf)
dN_dmqz = HMF.N_of_mqz_SZ(lndMDn*massMultiplier,qbin_edges,SZProf)

Nup = getNmzq(dUp,mexp_edges,z_edges,qbin_edges)        
Ndn = getNmzq(dDn,mexp_edges,z_edges,qbin_edges)
dNdp = (Nup-Ndn)/rayStep
np.save(bigDataDir+"Nup_mzq_"+saveId+"_sigR",Nup)
np.save(bigDataDir+"Ndn_mzq_"+saveId+"_sigR",Ndn)
np.save(bigDataDir+"dNdp_mzq_"+saveId+"_sigR",dNdp)





    
