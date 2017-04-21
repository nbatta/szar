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


if debug: print "Starting common module imports..."

from mpi4py import MPI
from szlib.szcounts import ClusterCosmology,Halo_MF,SZ_Cluster_Model,getNmzq
import numpy as np
    
if debug: print "Finished common module imports."

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()    




# the boss prepares cosmology objects for the minions
# Also, I really don't want all my cores to import a bunch of
# python modules
if rank==0:

    if debug: print "Starting rank 0 imports..."

    import sys
    from ConfigParser import SafeConfigParser 
    import cPickle as pickle

    if debug: print "Finished rank 0 imports. Starting rank 0 work..."
    

    expName = sys.argv[1]
    gridName = sys.argv[2]
    calName = sys.argv[3]

    # Let's read in all parameters that can be varied by looking
    # for those that have step sizes specified. All the others
    # only have fiducials.
    iniFile = "input/pipeline.ini"
    Config = SafeConfigParser()
    Config.optionxform=str
    Config.read(iniFile)
    bigDataDir = Config.get('general','bigDataDirectory')


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

    assert numcores==3, "I need 3 cores to do my job for 1 params. \
    You gave me "+str(numcores)+ " core(s) for 1 param(s)."

    version = Config.get('general','version')

    calFileUp = bigDataDir+"lensgridRayUp_"+expName+"_"+gridName+"_"+calName+ "_v" + version+".pkl"
    calFileDn = bigDataDir+"lensgridRayDn_"+expName+"_"+gridName+"_"+calName+ "_v" + version+".pkl"
    
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
    if debug: print "Finished rank 0 work."

else:
    rayStep = None
    fparams = None
    mexp_edges = None
    z_edges = None
    lndMUp = None
    lndMDn = None
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

if rank==0: print "Broadcasting..."
rayStep = comm.bcast(rayStep, root = 0)
fparams = comm.bcast(fparams, root = 0)
mexp_edges = comm.bcast(mexp_edges, root = 0)
z_edges = comm.bcast(z_edges, root = 0)
lndMUp = comm.bcast(lndMUp, root = 0)
lndMDn = comm.bcast(lndMDn, root = 0)
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
if rank==0: print "Broadcasted."

myParamIndex = (rank+1)/2-1
passParams = fparams.copy()

    

cc = ClusterCosmology(passParams,constDict,clTTFixFile=clttfile)
HMF = Halo_MF(cc,mexp_edges,z_edges)
HMF.sigN = siggrid.copy()
SZProf = SZ_Cluster_Model(cc,clusterDict,rms_noises = noise,fwhms=beam,freqs=freq,lknee=lknee,alpha=alpha)
if rank==1:
    dN_dmqz = HMF.N_of_mqz_SZ(lndMUp*massMultiplier,qbin_edges,SZProf)
elif rank==2:
    dN_dmqz = HMF.N_of_mqz_SZ(lndMDn*massMultiplier,qbin_edges,SZProf)

if rank==0: 

    print "Waiting for ups and downs..."
    for i in range(1,numcores):
        data = np.empty((mexp_edges.size-1,z_edges.size-1,qbin_edges.size-1), dtype=np.float64)
        comm.Recv(data, source=i, tag=77)
        if i==1:
            dUp = data.copy()
        elif i==2:
            dDn = data.copy()
            
            
    Nup = getNmzq(dUp,mexp_edges,z_edges,qbin_edges)        
    Ndn = getNmzq(dDn,mexp_edges,z_edges,qbin_edges)
    dNdp = (Nup-Ndn)/rayStep
    np.save(bigDataDir+"Nup_mzq_"+saveId+"_sigR",Nup)
    np.save(bigDataDir+"Ndn_mzq_"+saveId+"_sigR",Ndn)
    np.save(bigDataDir+"dNdp_mzq_"+saveId+"_sigR",dNdp)
    
else:
    data = dN_dmqz.astype(np.float64)
    comm.Send(data, dest=0, tag=77)




    
