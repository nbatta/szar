import matplotlib
matplotlib.use('Agg')
from mpi4py import MPI
import numpy as np
from szlib.szcounts import ClusterCosmology,Halo_MF,SZ_Cluster_Model

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()    



if rank==0:

    import sys
    from ConfigParser import SafeConfigParser 
    import cPickle as pickle

    expName = sys.argv[1]
    calName = sys.argv[2]


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

    from orphics.tools.io import dictFromSection, listFromConfig

    ms = listFromConfig(Config,calName,'mexprange')
    mgrid = np.arange(ms[0],ms[1],ms[2])
    zs = listFromConfig(Config,calName,'zrange')
    zgrid = np.arange(zs[0],zs[1],zs[2])

    beam = listFromConfig(Config,expName,'beams')
    noise = listFromConfig(Config,expName,'noises')
    freq = listFromConfig(Config,expName,'freqs')
    lkneeT,lkneeP = listFromConfig(Config,expName,'lknee')
    alphaT,alphaP = listFromConfig(Config,expName,'alpha')
    lmax = int(Config.getfloat(expName,'lmax'))
    lknee = lkneeT
    alpha = alphaT

    clttfile = Config.get('general','clttfile')
       

    clusterDict = dictFromSection(Config,'cluster_params')

    constDict = dictFromSection(Config,'constants')

else:
    mgrid = None
    zgrid = None
    fparams = None
    constDict = None
    clusterDict = None
    clttfile = None
    beam = None
    noise = None
    freq = None
    lknee = None
    alpha = None

if rank==0: print "Broadcasting..."
mgrid = comm.bcast(mgrid, root = 0)
zgrid = comm.bcast(zgrid, root = 0)
fparams = comm.bcast(fparams, root = 0)
constDict = comm.bcast(constDict, root = 0)
clusterDict = comm.bcast(clusterDict, root = 0)
clttfile = comm.bcast(clttfile, root = 0)
beam = comm.bcast(beam, root = 0)
noise = comm.bcast(noise, root = 0)
freq = comm.bcast(freq, root = 0)
lknee = comm.bcast(lknee, root = 0)
alpha = comm.bcast(alpha, root = 0)
if rank==0: print "Broadcasted."


# print ls,Nls
cc = ClusterCosmology(fparams,constDict,clTTFixFile=clttfile)
HMF = Halo_MF(cc,mgrid,zgrid)
SZCluster = SZ_Cluster_Model(cc,clusterDict,rms_noises = noise,fwhms=beam,freqs=freq,lknee=lknee,alpha=alpha)

numms = mgrid.size
numzs = zgrid.size
siggrid = np.zeros((numms,numzs))
numes = siggrid.size



indices = np.arange(numes)
splits = np.array_split(indices,numcores)

if rank==0:
    lengths = [len(split) for split in splits]
    mintasks = min(lengths)
    maxtasks = max(lengths)
    print "I have ",numcores, " cores to work with."
    print "And I have ", numes, " tasks to do."
    print "Each worker gets at least ", mintasks, " tasks and at most ", maxtasks, " tasks."
    buestguess = 0.5*maxtasks
    print "My best guess is that this will take ", buestguess, " seconds."
    print "Starting the slow part..."


mySplitIndex = rank

mySplit = splits[mySplitIndex]

for index in mySplit:
            
    mindex,zindex = np.unravel_index(index,siggrid.shape)
    mass = mgrid[mindex]
    z = zgrid[zindex]

    var = SZCluster.quickVar(mass,z)
    siggrid[mindex,zindex] = np.sqrt(var)
                
    


if rank!=0:
    siggrid = siggrid.astype(np.float64)
    comm.Send(siggrid, dest=0, tag=77)
else:
    print "Waiting for workers..."
    for i in range(1,numcores):
        data = np.zeros(siggrid.shape, dtype=np.float64)
        comm.Recv(data, source=i, tag=77)
        siggrid += data
        

    import cPickle as pickle
    pickle.dump((mgrid,zgrid,siggrid),open("data/siggrid_"+expName+"_"+calName+".pkl",'wb'))
