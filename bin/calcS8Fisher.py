import matplotlib
matplotlib.use('Agg')
import numpy as np
import sys, os, time
from szar.counts import rebinN,getA
from orphics.tools.io import Plotter,dictFromSection,listFromConfig
from ConfigParser import SafeConfigParser 
import cPickle as pickle
from szar.fisher import getFisher
import szar.fisher as sfisher

expName = sys.argv[1]
gridName = sys.argv[2]
calName = sys.argv[3]


iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)
version = Config.get('general','version')
pzcutoff = Config.getfloat('general','photoZCutOff')



bigDataDir = Config.get('general','bigDataDirectory')
saveId = sfisher.save_id(expName,gridName,calName,version)
derivRoot = sfisher.deriv_root(bigDataDir,saveId)


from orphics.tools.io import dictFromSection, listFromConfig
fsky = Config.getfloat(expName,'fsky')



from orphics.tools.io import listFromConfig


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



# Fisher params
fishName = 'lcdm-paper'
fishSection = "fisher-"+fishName
origParams = Config.get(fishSection,'paramList').split(',')


saveName = Config.get(fishSection,'saveSuffix')
##########################
# Populate Fisher
Fisher, paramList = sfisher.cluster_fisher_from_config(Config,expName,gridName,calName,fishName,s8=True)
##########################



# Number of non-SZ params (params that will be in Planck/BAO)
numCosmo = Config.getint(fishSection,'numCosmo')
numLeft = len(paramList) - numCosmo


# pl = Plotter()
# pl.plot2d(cov2corr(fisherPlanck))
# pl.done("output/fisherplanck.png")
# sys.exit()


f = Fisher

pickle.dump((paramList,f),open(bigDataDir+"savedS8Fisher_"+saveId+"_"+saveName+".pkl",'wb'))



inv = np.linalg.inv(f)

err = np.sqrt(np.diagonal(inv))[len(origParams):]

print err

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
