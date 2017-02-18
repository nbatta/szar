"""

Calculates cluster count derivatives using MPI.

Always reads values from input/pipelineMakeDerivs.py, including
parameter fiducials and step-sizes.

python bin/makeDerivs.py <paramList> <expName> <calName>

<paramList> is comma separated param list, no spaces, case-sensitive.

If <paramList> is "allParams", calculates derivatives for all
params with step sizes in [params] section of ini file.

<expName> is name of section in input/pipelineMakeDerivs.py
that specifies an experiment.

<calName> is the name of a pickle file containing the mass
calibration error over mass.

"""

from mpi4py import MPI
import sys
from ConfigParser import SafeConfigParser 
import cPickle as pickle

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()    


inParamList = sys.argv[1].split(',')
expName = sys.argv[2]
calName = sys.argv[3]
calFile = sys.argv[4]


# the boss prepares cosmology objects for the minions
if rank==0:
    # Let's read in all parameters that can be varied by looking
    # for those that have step sizes specified. All the others
    # only have fiducials.
    iniFile = "input/pipelineMakeDerivs.ini"
    Config = SafeConfigParser()
    Config.optionxform=str
    Config.read(iniFile)

    paramList = [] # the parameters that can be varied
    fparams = {}   # the 
    stepSizes = {}
    for (key, val) in Config.items('params'):
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
            assert param in stepSizes.keys(), param + " not found in stepSizes dict. Looks like a bug in the code."
    

    numParams = len(inParamList)
    neededCores = 2*numParams+1
    assert numcores==neededCores, "I need 2N+1 cores to do my job for N params. \
    You gave me "+str(numcores)+ " core(s) for "+str(numParams)+" param(s)."


    # load the mass calibration grid
    mexprange, zrange, lndM = pickle.load(open(calFile,"rb"))

else:
    inParamList = None
    stepSizes = None
    fparams = None
    mexprange = None
    zrange = None
    lndM = None

inParamList = comm.bcast(inParamList, root = 0)
stepSizes = comm.bcast(stepSizes, root = 0)
fparams = comm.bcast(fparams, root = 0)
mexprange = comm.bcast(mexprange, root = 0)
zrange = comm.bcast(zrange, root = 0)
lndM = comm.bcast(lndM, root = 0)

myParamIndex = (rank+1)/2-1
passParams = fparams.copy()


# If boss, do the fiducial. If odd rank, the minion is doing an "up" job, else doing a "down" job
if rank ==0:
    myParam = "fid"
    upDown = ""
elif rank%2==1:
    myParam = inParamList[myParamIndex]
    passParams[myParam] = fparams[myParam] + stepSizes[myParam]/2.
    upDown = "_up"

elif rank%2==0:
    myParam = inParamList[myParamIndex]
    passParams[myParam] = fparams[myParam] - stepSizes[myParam]/2.
    upDown = "_down"

print rank,myParam,upDown
# cc = ClusterCosmology(passParams,constDict,lmax)
# HMF = Halo_MF(clusterCosmology=cc)
# dN_dmqz = HMFup.N_of_mqz_SZ(lndM,zrange,mexprange,np.exp(qbin),beam,noise,freq,clusterDict,lknee,alpha)
# saveId = expName + "_" + calName
# np.save("data/dN_dzmq_"+saveId+"_"+myParam+upDown,dN_dmqz)

    
