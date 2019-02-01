from __future__ import print_function
from __future__ import division
from builtins import zip
from builtins import range
from past.utils import old_div
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
parser.add_argument("Mmin", type=float,help='Minimum mass.')

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
pardict = nemoOutputDir + 'equD56.par'
noise_file = 'RMSMap_Arnaud_M2e14_z0p4.fits'

fixlist = ['omch2','ombh2','H0','ns','tau','massbias','yslope','scat']
fixvals = [0.1225,0.0245,70,0.97,0.06,1.0,0.08,0.2]
fix_params = dict(list(zip(fixlist,fixvals)))

parlist = ['As']
parvals = [2.0e-09]

start = time.time()
filedir = '/Users/nab/Desktop/Projects/ACTPol_Cluster_Like/'
filename = args.chain_name #,'mockCat_v1'
mmin = args.Mmin
buff = 0.2

MC = lk.MockCatalog(iniFile,pardict,nemoOutputDir,noise_file,parvals,parlist,mass_grid_log=[mmin,15.7,0.01],z_grid=[0.1,2.01,0.1])

MC2 = lk.MockCatalog(iniFile,pardict,nemoOutputDir,noise_file,parvals,parlist,mass_grid_log=[mmin+buff,15.7,0.01],z_grid=[0.1,2.01,0.1])

print("pre-predictions Ntot ", MC2.Total_clusters(MC2.fsky), "+/-", np.sqrt(MC2.Total_clusters(MC2.fsky)))

print("Actual cut",MC.test_Mockcat_Nums(mmin+buff))


MC3 = lk.MockCatalog(iniFile,pardict,nemoOutputDir,noise_file,parvals,parlist,mass_grid_log=[14.0,16.0,0.01],z_grid=[0.01,3.02,0.1])

print("pre-predictions Ntot ", MC3.Total_clusters_check(0.4), "+/-", np.sqrt(MC3.Total_clusters_check(0.4)))
