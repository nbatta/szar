import numpy as np
import szar.likelihood as lk
import matplotlib.pyplot as plt
from scipy import stats
from configparser import SafeConfigParser
from orphics import io
from orphics.io import Plotter
from szar.counts import ClusterCosmology,Halo_MF
from nemo import simsTools
from astropy.io import fits

import emcee
import time, sys, os
# from emcee.utils import MPIPool
import argparse
# Parse command line
parser = argparse.ArgumentParser(description='Run likelihood.')
parser.add_argument("chain_name", type=str,help='Root name of run.')
parser.add_argument("-t", "--testMock",     type=int,  default=0,help="number of mocks.")

args = parser.parse_args()

iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)
bigDataDir = Config.get('general','bigDataDirectory')
clttfile = Config.get('general','clttfile')
constDict = io.dict_from_section(Config,'constants')
version = Config.get('general','version')

fparams = {}
for (key, val) in Config.items('params'):
    if ',' in val:
        param, step = val.split(',')
        fparams[key] = float(param)
    else:
        fparams[key] = float(val)

PathConfig = io.load_path_config()
nemoOutputDir = PathConfig.get("likepaths","nemoOutputDir")
nemoOutputDirOut = PathConfig.get("likepaths","nemoOutputDirOut")
chain_out = PathConfig.get("likepaths","chains")

pardict = nemoOutputDir + 'equD56.par'
noise_file = 'RMSMap_Arnaud_M2e14_z0p4.fits'


fixlist = ['omch2','ombh2','H0','ns','tau','massbias','yslope','scat']
fixvals = [0.1225,0.0245,70,0.97,0.06,1.0,0.08,0.2]
fix_params = dict(zip(fixlist,fixvals))

parlist = ['As']
parvals = [2.0e-09]
mmin = 14.3
print mmin

priorlist = []
prioravg = np.array([])
priorwth = np.array([])
priorvals = np.array([prioravg,priorwth])

parlist_cat = np.append(fixlist, parlist) 
parvals_cat = np.append(fixvals,parvals)
#print parlist_cat
#print parvals_cat
    
start = time.time()
filedir = '/Users/nab/Desktop/Projects/ACTPol_Cluster_Like/'
filename = args.chain_name #,'mockCat_v1'

if args.testMock:
    print "Testing Mockcat Numbers"
    MC = lk.MockCatalog(iniFile,pardict,nemoOutputDir,noise_file,parvals,parlist,mass_grid_log=[mmin-0.1,15.7,0.01],z_grid=[0.1,2.01,0.1])
    saveNum = []
    for i in xrange(args.testMock):
        Nums = MC.test_Mockcat_Nums(mmin)
        saveNum = np.append(saveNum,Nums)
        if (np.mod(i,args.testMock/10) == 0):
            print "."
    #print saveNum
    f = open(filedir+filename+'.txt', "w")
    np.savetxt(f,saveNum)
    print ('sample time',time.time() - start)    
    sys.exit(0)
else:
    check = os.path.isfile(filedir+filename+'.fits')
    if (check == False):
        MC = lk.MockCatalog(iniFile,pardict,nemoOutputDir,noise_file,parvals,parlist,mass_grid_log=[mmin-0.1,15.7,0.01],z_grid=[0.1,2.01,0.1])
        MC.write_test_cat_toFits(filedir,filename)
    #MC.write_obs_cat_toFits(filedir,filename)
    else:
        print "Mockcat already exists"

print ('sample time',time.time() - start)    


MC2 = lk.MockCatalog(iniFile,pardict,nemoOutputDir,noise_file,parvals,parlist,mass_grid_log=[mmin,15.7,0.01],z_grid=[0.1,2.01,0.1])

print "pre-predictions Ntot ", MC2.Total_clusters(MC2.fsky)   
#list = fits.open(filedir+filename+'.fits')
#if list:
#    print "Using file", filename+'.fits'
#else:
#

start = time.time()

filename_out = chain_out+"/test_likelival_"+args.chain_name+".dat"

TL = lk.clustLikeTest(iniFile,filedir+filename,fix_params,mmin=mmin)

#print TL.mgrid

parvals_arr = parvals*(1+np.arange(-0.1,0.1001,0.05))
ansout = parvals_arr*0.0
for ii, vals in enumerate(parvals_arr):
        #print ii, vals
    ansout[ii] = TL.lnlike([vals],parlist)
    
f = open(filename, "w")
savemat = [parvals_arr,ansout]
np.savetxt(f,savemat)

indmin = np.argmax(ansout)
print parvals_arr[indmin]

sys.exit(0)

print (time.time() - start)  
