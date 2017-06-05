import matplotlib
matplotlib.use('Agg')
import numpy as np
import sys, os, time
from szar.counts import rebinN,getA
from orphics.tools.io import Plotter,dictFromSection,listFromConfig
from ConfigParser import SafeConfigParser 
import cPickle as pickle
from szar.fisher import getFisher


expName = sys.argv[1]
gridName = sys.argv[2]
calName = sys.argv[3]


iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)
version = Config.get('general','version')
pzcutoff = Config.getfloat('general','photoZCutOff')

fparams = {}   # the 
for (key, val) in Config.items('params'):
    if ',' in val:
        param, step = val.split(',')
        fparams[key] = float(param)
    else:
        fparams[key] = float(val)



bigDataDir = Config.get('general','bigDataDirectory')

saveId = expName + "_" + gridName + "_" + calName + "_v" + version

from orphics.tools.io import dictFromSection, listFromConfig
fsky = Config.getfloat(expName,'fsky')

# get s/n q-bins
qs = listFromConfig(Config,'general','qbins')
qspacing = Config.get('general','qbins_spacing')
if qspacing=="log":
    qbins = np.logspace(np.log10(qs[0]),np.log10(qs[1]),int(qs[2])+1)
elif qspacing=="linear":
    qbins = np.linspace(qs[0],qs[1],int(qs[2])+1)
else:
    raise ValueError



from orphics.tools.io import listFromConfig
zs = listFromConfig(Config,gridName,'zrange')
z_edges = np.arange(zs[0],zs[1]+zs[2],zs[2])
zrange = (z_edges[1:]+z_edges[:-1])/2.


NFid_mzq = np.load(bigDataDir+"N_mzq_"+saveId+"_fid_sigma8.npy")
NFid_mzq_alt = np.load(bigDataDir+"N_mzq_"+saveId+"_fid.npy")
try:
    assert np.all(np.isclose(NFid_mzq,NFid_mzq_alt))
except:
    print "ERROR: Sigma8 and fid saves are not identical"
    print NFid_mzq.shape
    print NFid_mzq_alt.shape
    from orphics.tools.io import quickPlot2d
    quickPlot2d(NFid_mzq[0,:,:],os.environ['WWW']+"debug_sig8.png")
    quickPlot2d(NFid_mzq_alt[0,:,:],os.environ['WWW']+"debug_fid.png")
    sys.exit(1)

    
import itertools
new_z_edges, N_fid = rebinN(NFid_mzq,pzcutoff,z_edges)
N_fid = N_fid*fsky
print "Total number of clusters: ", N_fid.sum() #getTotN(N_fid,mgrid,zgrid,qbins)



# Fisher params
fishSection = 'fisher-lcdm'
zlist = ["S8Z"+str(i) for i in range(len(zrange))]
origParams = Config.get(fishSection,'paramList').split(',')
paramList = origParams+zlist
saveName = Config.get(fishSection,'saveSuffix')
numParams = len(paramList)
Fisher = np.zeros((numParams,numParams))
paramCombs = itertools.combinations_with_replacement(paramList,2)


sId = expName + "_" + gridName + "_v" + version
#sovernsquareEach = np.loadtxt(bigDataDir+"sampleVarGrid_"+sId+".txt")
#sovernsquare =  np.dstack([sovernsquareEach]*(len(qbins)-1))


try:
    priorNameList = Config.get(fishSection,'prior_names').split(',')
    priorValueList = listFromConfig(Config,fishSection,'prior_values')
except:
    priorNameList = []
    priorValueList = []
    
##########################
# Populate Fisher
Fisher = getFisher(N_fid,paramList,priorNameList,priorValueList,bigDataDir,saveId,pzcutoff,z_edges,fsky)
##########################


# Populate Fisher
# for param1,param2 in paramCombs:
#     if param1=='tau' or param2=='tau': continue
#     print param1,param2
    
#     new_z_edges, dN1 = rebinN(np.load(bigDataDir+"dNdp_mzq_"+saveId+"_"+param1+".npy"),pzcutoff,z_edges)
#     new_z_edges, dN2 = rebinN(np.load(bigDataDir+"dNdp_mzq_"+saveId+"_"+param2+".npy"),pzcutoff,z_edges)
#     dN1 = dN1[:,:,:]*fsky
#     dN2 = dN2[:,:,:]*fsky


#     i = paramList.index(param1)
#     j = paramList.index(param2)

#     assert not(np.any(np.isnan(dN1)))
#     assert not(np.any(np.isnan(dN2)))
#     assert not(np.any(np.isnan(N_fid)))


#     with np.errstate(divide='ignore'):
#         FellBlock = dN1*dN2*np.nan_to_num(1./(N_fid))#+(N_fid*N_fid*sovernsquare)))

#     Fell = FellBlock.sum()
        
       
#     Fisher[i,j] = Fell
#     Fisher[j,i] = Fell    



# Planck and BAO Fishers
planckFile = Config.get(fishSection,'planckFile')
try:
    baoFile = Config.get(fishSection,'baoFile')
except:
    baoFile = ''

# Number of non-SZ params (params that will be in Planck/BAO)
numCosmo = Config.getint(fishSection,'numCosmo')
numLeft = len(paramList) - numCosmo

fisherPlanck = 0.
if planckFile!='':
    try:
        fisherPlanck = np.loadtxt(planckFile)
    except:
        fisherPlanck = np.loadtxt(planckFile,delimiter=',')
    fisherPlanck = np.pad(fisherPlanck,pad_width=((0,numLeft),(0,numLeft)),mode="constant",constant_values=0.)

if baoFile!='':
    raise NotImplementedError



# pl = Plotter()
# pl.plot2d(cov2corr(fisherPlanck))
# pl.done("output/fisherplanck.png")
# sys.exit()


f = Fisher+fisherPlanck #[11:,11:]#+fisherPlanck

pickle.dump((paramList,f),open(bigDataDir+"savedS8Fisher_"+saveId+"_"+saveName+".pkl",'wb'))



# inv = np.linalg.inv(f)

# err = np.sqrt(np.diagonal(inv))[len(origParams):]

# print err

# import camb
# from szar.counts import ClusterCosmology

# constDict = dictFromSection(Config,'constants')


# s8,As1 = getA(fparams,constDict,zrange)
# print s8
# fparams['w0']=-0.7
# s8,As2 = getA(fparams,constDict,zrange)
# print s8

# outDir = "/gpfs01/astro/www/msyriac/"




# pl = Plotter(labelX="$z$",labelY="$\sigma_8(z)/\sigma_8(0)$",ftsize=12)
# pl.addErr(zrange,As1,yerr=err,xerr=np.diff(z_edges)/2.)
# pl.addErr(zrange,As2,yerr=err,xerr=np.diff(z_edges)/2.)
# #pl.legendOn()
# #pl._ax.set_ylim(0.9,1.1)
# pl.done(outDir+"s8errsowl.png")


# from orphics.tools.stats import cov2corr
# corr = cov2corr(f)    
# pl = Plotter()
# pl.plot2d(corr)
# pl.done(outDir+"corrowl.png")
