"""

Calculates cluster count derivatives for w_a using MPI.

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

from mpi4py import MPI
from szlib.szcounts import ClusterCosmology,Halo_MF
import numpy as np
    

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()    

assert numcores==3


# the boss prepares cosmology objects for the minions
# Also, I really don't want all my cores to import a bunch of
# python modules
if rank==0:

    import sys
    from ConfigParser import SafeConfigParser 
    import cPickle as pickle

    expName = sys.argv[1]
    calName = sys.argv[2]
    calFile = sys.argv[3]
    powerFile = sys.argv[4]
    stepSize = float(sys.argv[5])

    iniFile = "input/pipeline.ini"
    Config = SafeConfigParser()
    Config.optionxform=str
    Config.read(iniFile)

    fparams = {} 
    for (key, val) in Config.items('params'):
        if ',' in val:
            param, step = val.split(',')
            fparams[key] = float(param)
        else:
            fparams[key] = float(val)




    numParams = 1


    suffix = Config.get('general','suffix')
    # load the mass calibration grid
    mexprange, zrange, lndM = pickle.load(open(calFile,"rb"))


    zrange = np.insert(zrange,0,0.0)
    saveId = expName + "_" + calName + "_" + suffix

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
        qbins = np.logspace(np.log10(qs[0]),np.log10(qs[1]),int(qs[2]))
    elif qspacing=="linear":
        qbins = np.linspace(qs[0],qs[1],int(qs[2]))
    else:
        raise ValueError

    massMultiplier = Config.getfloat('general','mass_calib_factor')

else:
    fparams = None
    mexprange = None
    zrange = None
    lndM = None
    saveId = None
    constDict = None
    clttfile = None
    qbins = None
    clusterDict = None
    beam = None
    noise = None
    freq = None
    lknee = None
    alpha = None
    massMultiplier = None
    powerFile = None

if rank==0: print "Broadcasting..."
fparams = comm.bcast(fparams, root = 0)
mexprange = comm.bcast(mexprange, root = 0)
zrange = comm.bcast(zrange, root = 0)
lndM = comm.bcast(lndM, root = 0)
saveId = comm.bcast(saveId, root = 0)
constDict = comm.bcast(constDict, root = 0)
clttfile = comm.bcast(clttfile, root = 0)
qbins = comm.bcast(qbins, root = 0)
clusterDict = comm.bcast(clusterDict, root = 0)
beam = comm.bcast(beam, root = 0)
noise = comm.bcast(noise, root = 0)
freq = comm.bcast(freq, root = 0)
lknee = comm.bcast(lknee, root = 0)
alpha = comm.bcast(alpha, root = 0)
massMultiplier = comm.bcast(massMultiplier, root = 0)
powerFile = comm.bcast(powerFile, root = 0)
if rank==0: print "Broadcasted."

passParams = fparams.copy()


# If boss, do the fiducial. If odd rank, the minion is doing an "up" job, else doing a "down" job
if rank==0:
    override = None
elif rank%2==1:
    override = powerFile + "Up"
elif rank%2==0:
    override = powerFile + "Dn"

    

if rank==0: 

    print "Waiting for ups and downs..."
    for i in range(1,numcores):
        data = np.empty((mexprange.size,zrange.size-1,qbins.size), dtype=np.float64)
        comm.Recv(data, source=i, tag=77)
        if i%2==1:
            dUps = data.copy()
        elif i%2==0:
            dDns = data.copy()

    dN = (dUps-dDns)/stepSize
    np.save("data/dNup_dzmq_"+saveId+"_wa",dUps)
    np.save("data/dNdn_dzmq_"+saveId+"_wa",dDns)
    np.save("data/dN_dzmq_"+saveId+"_wa",dN)
        
else:
    cc = ClusterCosmology(passParams,constDict,clTTFixFile=clttfile)
    HMF = Halo_MF(clusterCosmology=cc,overridePowerSpectra=override)
    dN_dmqz = HMF.N_of_mqz_SZ(lndM*massMultiplier,zrange,mexprange,qbins,beam,noise,freq,clusterDict,lknee,alpha)
    data = dN_dmqz.astype(np.float64)
    comm.Send(data, dest=0, tag=77)




    
