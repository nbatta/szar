"""

Calculates cluster count derivatives using MPI.

Always reads values from input/pipelineMakeDerivs.py, including
parameter fiducials and step-sizes.

python bin/makeDerivs.py <paramList> <expName> <calName> <calFile>

<paramList> is comma separated param list, no spaces, case-sensitive.

If <paramList> is "allParams", calculates derivatives for all
params with step sizes in [params] section of ini file.

<expName> is name of section in input/pipelineMakeDerivs.py
that specifies an experiment.

<calName> name of calibration that will be used in the saved files

<calFile> is the name of a pickle file containing the mass
calibration error over mass.

"""

debug = False


if debug: print("Starting common module imports...")

from mpi4py import MPI
from szar.counts import ClusterCosmology,Halo_MF,getNmzq
from szar.szproperties import SZ_Cluster_Model
import numpy as np
    
if debug: print("Finished common module imports.")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()    




# the boss prepares cosmology objects for the minions
# Also, I really don't want all my cores to import a bunch of
# python modules
if rank==0:

    if debug: print("Starting rank 0 imports...")

    import sys
    from configparser import SafeConfigParser 
    import pickle as pickle

    if debug: print("Finished rank 0 imports. Starting rank 0 work...")
    

    expName = sys.argv[1]
    gridName = sys.argv[2]
    calName = sys.argv[3]
    calFile = sys.argv[4]

    # Let's read in all parameters that can be varied by looking
    # for those that have step sizes specified. All the others
    # only have fiducials.
    iniFile = "input/pipeline.ini"
    Config = SafeConfigParser()
    Config.optionxform=str
    Config.read(iniFile)
    bigDataDir = Config.get('general','bigDataDirectory')
    waDerivRoot = bigDataDir+Config.get('general','waDerivRoot')


    fparams = {}   # the 
    stepSizes = {}
    for (key, val) in Config.items('params'):
        if ',' in val:
            param, step = val.split(',')
            fparams[key] = float(param)
            stepSizes[key] = float(step)
        else:
            fparams[key] = float(val)




    waStep = stepSizes['wa']

    assert numcores==3, "I need 3 cores to do my job for 1 params. \
    You gave me "+str(numcores)+ " core(s) for 1 param(s)."


    version = Config.get('general','version')
    # load the mass calibration grid
    mexp_edges, z_edges, lndM = pickle.load(open(calFile,"rb"))

    mgrid,zgrid,siggrid = pickle.load(open(bigDataDir+"szgrid_"+expName+"_"+gridName+ "_v" + version+".pkl",'rb'))
    assert np.all(mgrid==mexp_edges)
    assert np.all(z_edges==zgrid)
    
    saveId = expName + "_" + gridName + "_" + calName + "_v" + version

    from orphics.io import dict_from_section, list_from_config
    constDict = dict_from_section(Config,'constants')
    clusterDict = dict_from_section(Config,'cluster_params')
    beam = list_from_config(Config,expName,'beams')
    noise = list_from_config(Config,expName,'noises')
    freq = list_from_config(Config,expName,'freqs')
    lknee = list_from_config(Config,expName,'lknee')[0]
    alpha = list_from_config(Config,expName,'alpha')[0]

    clttfile = Config.get('general','clttfile')

    # get s/n q-bins
    qs = list_from_config(Config,'general','qbins')
    qspacing = Config.get('general','qbins_spacing')
    if qspacing=="log":
        qbin_edges = np.logspace(np.log10(qs[0]),np.log10(qs[1]),int(qs[2])+1)
    elif qspacing=="linear":
        qbin_edges = np.linspace(qs[0],qs[1],int(qs[2])+1)
    else:
        raise ValueError

    massMultiplier = Config.getfloat('general','mass_calib_factor')
    if debug: print("Finished rank 0 work.")

else:
    waDerivRoot = None
    waStep = None
    fparams = None
    mexp_edges = None
    z_edges = None
    lndM = None
    saveId = None
    constDict = None
    clttfile = None
    qbin_edges = None
    clusterDict = None
    beam = None
    noise = None
    freq = None
    lknee = None
    alpha = None
    massMultiplier = None
    siggrid = None

if rank==0: print("Broadcasting...")
waDerivRoot = comm.bcast(waDerivRoot, root = 0)
waStep = comm.bcast(waStep, root = 0)
fparams = comm.bcast(fparams, root = 0)
mexp_edges = comm.bcast(mexp_edges, root = 0)
z_edges = comm.bcast(z_edges, root = 0)
lndM = comm.bcast(lndM, root = 0)
saveId = comm.bcast(saveId, root = 0)
constDict = comm.bcast(constDict, root = 0)
clttfile = comm.bcast(clttfile, root = 0)
qbin_edges = comm.bcast(qbin_edges, root = 0)
clusterDict = comm.bcast(clusterDict, root = 0)
beam = comm.bcast(beam, root = 0)
noise = comm.bcast(noise, root = 0)
freq = comm.bcast(freq, root = 0)
lknee = comm.bcast(lknee, root = 0)
alpha = comm.bcast(alpha, root = 0)
massMultiplier = comm.bcast(massMultiplier, root = 0)
siggrid = comm.bcast(siggrid, root = 0)
if rank==0: print("Broadcasted.")

myParamIndex = (rank+1)/2-1
passParams = fparams.copy()

    
if rank==1:
    fileSuff = "Up"
elif rank==2:
    fileSuff = "Dn"

if rank!=0:    
    pFile = lambda z: waDerivRoot+str(waStep)+fileSuff+"_matterpower_"+str(z)+".dat"
    zcents = (z_edges[1:]+z_edges[:-1])/2.
    for inum,z in enumerate(zcents):
        kh,p = np.loadtxt(pFile(z),unpack=True)
        if inum==0:
            khorig = kh.copy()
            pk = np.zeros((zcents.size,kh.size))
        assert np.all(np.isclose(kh,khorig))
        pk[inum,:] = p.copy()
else:
    kh = None
    pk = None

cc = ClusterCosmology(passParams,constDict,clTTFixFile=clttfile)
HMF = Halo_MF(cc,mexp_edges,z_edges,kh=kh,powerZK=pk)
HMF.sigN = siggrid.copy()
SZProf = SZ_Cluster_Model(cc,clusterDict,rms_noises = noise,fwhms=beam,freqs=freq,lknee=lknee,alpha=alpha)
dN_dmqz = HMF.N_of_mqz_SZ(lndM*massMultiplier,qbin_edges,SZProf)


if rank==0: 
    np.save(bigDataDir+"N_mzq_"+saveId+"_wa_fid",getNmzq(dN_dmqz,mexp_edges,z_edges,qbin_edges))

    print("Waiting for ups and downs...")
    for i in range(1,numcores):
        data = np.empty(dN_dmqz.shape, dtype=np.float64)
        comm.Recv(data, source=i, tag=77)
        if i==1:
            dUp = data.copy()
        elif i==2:
            dDn = data.copy()
            
            
    Nup = getNmzq(dUp,mexp_edges,z_edges,qbin_edges)        
    Ndn = getNmzq(dDn,mexp_edges,z_edges,qbin_edges)
    dNdp = (Nup-Ndn)/waStep
    np.save(bigDataDir+"Nup_mzq_"+saveId+"_wa",Nup)
    np.save(bigDataDir+"Ndn_mzq_"+saveId+"_wa",Ndn)
    np.save(bigDataDir+"dNdp_mzq_"+saveId+"_wa",dNdp)
    
else:
    data = dN_dmqz.astype(np.float64)
    comm.Send(data, dest=0, tag=77)




    
