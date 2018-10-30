"""

Calculates clustering derivatives using MPI.

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
from __future__ import print_function
from __future__ import division

from builtins import str
from builtins import range
from past.utils import old_div
debug = False


if debug: print("Starting common module imports...")

from mpi4py import MPI
from szar.clustering import Clustering
from szar.szproperties import SZ_Cluster_Model
import szar.fisher as sfisher
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
    

    inParamList = sys.argv[1].split(',')
    expName = sys.argv[2]
    gridName = sys.argv[3]
    calName = sys.argv[4]
    calFile = sys.argv[5]

    # Let's read in all parameters that can be varied by looking
    # for those that have step sizes specified. All the others
    # only have fiducials.
    iniFile = "input/pipeline.ini"
    Config = SafeConfigParser()
    Config.optionxform=str
    Config.read(iniFile)
    bigDataDir = Config.get('general','bigDataDirectory')

    manualParamList = Config.get('general','manualParams').split(',')

    paramList = [] # the parameters that can be varied
    fparams = {}   # the 
    stepSizes = {}
    for (key, val) in Config.items('params'):
        if key in manualParamList: continue
        if ',' in val:
            param, step = val.split(',')
            paramList.append(key)
            fparams[key] = float(param)
            stepSizes[key] = float(step)
        else:
            fparams[key] = float(val)



    if inParamList[0]=="allParams":
        assert len(inParamList)==1, "I'm confused why you'd specify more params with allParams."
        
        inParamList = paramList

    else:
        for param in inParamList:
            assert param in paramList, param + " not found in ini file with a specified step size."
            assert param in list(stepSizes.keys()), param + " not found in stepSizes dict. Looks like a bug in the code."



    print(paramList)

    numParams = len(inParamList)
    neededCores = 2*numParams+1
    try:
        assert numcores==neededCores, "I need 2N+1 cores to do my job for N params. \
        You gave me "+str(numcores)+ " core(s) for "+str(numParams)+" param(s)."
    except:
        print(inParamList)
        sys.exit()


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
    fsky = Config.getfloat(expName,'fsky')
    try:
        v3mode = Config.getint(expName,'V3mode')
    except:
        v3mode = -1


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
    YWLcorrflag = Config.getfloat('general','ywl_corr_flag')

    if debug: print("Finished rank 0 work.")

else:
    inParamList = None
    stepSizes = None
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
    fsky = None
    massMultiplier = None
    siggrid = None
    YWLcorrflag = None
    v3mode = None

if rank==0: print("Broadcasting...")
inParamList = comm.bcast(inParamList, root = 0)
stepSizes = comm.bcast(stepSizes, root = 0)
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
fsky = comm.bcast(fsky, root = 0)
massMultiplier = comm.bcast(massMultiplier, root = 0)
siggrid = comm.bcast(siggrid, root = 0)
YWLcorrflag = comm.bcast(YWLcorrflag, root = 0)
v3mode = comm.bcast(v3mode, root = 0)
if rank==0: print("Broadcasted.")

myParamIndex = old_div((rank+1),2)-1
passParams = fparams.copy()


# If boss, do the fiducial. If odd rank, the minion is doing an "up" job, else doing a "down" job
if rank==0:
    pass
elif rank%2==1:
    myParam = inParamList[myParamIndex]
    passParams[myParam] = fparams[myParam] + old_div(stepSizes[myParam],2.)
elif rank%2==0:
    myParam = inParamList[myParamIndex]
    passParams[myParam] = fparams[myParam] - old_div(stepSizes[myParam],2.)


if rank!=0: print(rank,myParam,fparams[myParam],passParams[myParam])

####FIX THIS

cc = ClusterCosmology(passParams,constDict,clTTFixFile=clttfile)
clst = Clustering(expName,gridName,version,cc)

pbar = clst.fine_ps_bar(mu_grid)
veff = clst.V_eff(mu_grid)

fish_fac_err = veff * clst.HMF.kh**2 / pbar**2 # Tegmark 1997

#HMF = Halo_MF(cc,mexp_edges,z_edges)
#HMF.sigN = siggrid.copy()
#SZProf = SZ_Cluster_Model(cc,clusterDict,rms_noises = noise,fwhms=beam,freqs=freq,lknee=lknee,alpha=alpha,v3mode=v3mode,fsky=fsky)

#if (YWLcorrflag == 1):
#    dN_dmqz = HMF.N_of_mqz_SZ_corr(lndM*massMultiplier,qbin_edges,SZProf)
#else:
#dN_dmqz = HMF.N_of_mqz_SZ(lndM*massMultiplier,qbin_edges,SZProf)



if rank==0: 
    #np.save(bigDataDir+"N_dzmq_"+saveId+"_fid",dN_dmqz)
    np.save(sfisher.fid_file(bigDataDir,saveId),fish_fac_err)
    dUps = {}
    dDns = {}

    print("Waiting for ups and downs...")
    for i in range(1,numcores):
        data = np.empty(pbar.shape, dtype=np.float64)
        comm.Recv(data, source=i, tag=77)
        myParamIndex = old_div((i+1),2)-1
        if i%2==1:
            dUps[inParamList[myParamIndex]] = data.copy()
        elif i%2==0:
            dDns[inParamList[myParamIndex]] = data.copy()

    for param in inParamList:
        # dN = (dUps[param]-dDns[param])/stepSizes[param]
        # np.save(bigDataDir+"dNup_dzmq_"+saveId+"_"+param,dUps[param])
        # np.save(bigDataDir+"dNdn_dzmq_"+saveId+"_"+param,dDns[param])
        # np.save(bigDataDir+"dN_dzmq_"+saveId+"_"+param,dN)
        
        psbarup = dUps[param]
        psbardn = dDns[param]
        dpsbardp = (psbarup-psbardn)/stepSizes[param]
        np.save(bigDataDir+"Nup_mzq_"+saveId+"_"+param,psbarup)
        np.save(bigDataDir+"Ndn_mzq_"+saveId+"_"+param,psbardn)
        np.save(sfisher.deriv_root(bigDataDir,saveId)+param,dpsbardp)
        
else:
    data = psbar.astype(np.float64)
    comm.Send(data, dest=0, tag=77)




    
