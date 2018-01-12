import numpy as np
import szar.likelihood as lk
import matplotlib.pyplot as plt
from scipy import stats
from configparser import SafeConfigParser
from orphics import io
from orphics.io import Plotter
from szar.counts import ClusterCosmology,Halo_MF
from nemo import simsTools

import emcee
import time, sys, os
# from emcee.utils import MPIPool
import argparse
# Parse command line
parser = argparse.ArgumentParser(description='Run likelihood.')
parser.add_argument("chain_name", type=str,help='Root name of run.')
parser.add_argument("-i", "--index",     type=int,  default=0,help="Index of chainset.")
parser.add_argument("-N", "--nruns",     type=int,  default=int(1e6),help="Number of iterations.")
parser.add_argument("-t", "--test", action='store_true',help='Do a test quickly by setting Ntot=60 and just 3 params.')
parser.add_argument("-s", "--simtest", action='store_true',help='Do a test quickly by setting Ntot=60 and just 1 params.')
parser.add_argument("-p", "--printtest", action='store_true',help='Do quick print tests of likelihood functions.')
args = parser.parse_args()

# index = int(sys.argv[1])
print "Index ", args.index
index = args.index

iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)
bigDataDir = Config.get('general','bigDataDirectory')
clttfile = Config.get('general','clttfile')
constDict = io.dict_from_section(Config,'constants')
version = Config.get('general','version')
expName = "S4-1.0-CDT"#"S4-1.5-paper" #S4-1.0-CDT"
gridName = "grid-owl2" #grid-owl2"
#_S4-1.5-paper_grid-owl2_v0.6.p

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
# nemoOutputDir = '/gpfs01/astro/workarea/msyriac/data/depot/SZruns/ACTdata/' #/Users/nab/Desktop/Projects/ACTPol_Cluster_Like/ACTdata/'
# nemoOutputDirOut = '/gpfs01/astro/workarea/msyriac/data/depot/SZruns/ACTdata_out/'

pardict = nemoOutputDir + 'equD56.par'
noise_file = 'RMSMap_Arnaud_M2e14_z0p4.fits'


if args.test:
    fixlist = ['ombh2','ns','tau','massbias','yslope','scat']
    fixvals = [0.022,0.96,0.06,0.80,0.08,0.2]
elif args.simtest:
    fixlist = ['omch2','ombh2','H0','ns','tau','massbias','yslope','scat']
    fixvals = [0.1225,0.0245,70,0.97,0.06,1.0,0.08,0.2]
else:
    fixlist = ['tau']
    fixvals = [0.06]

fix_params = dict(zip(fixlist,fixvals))


CL = lk.clusterLike(iniFile,expName,gridName,pardict,nemoOutputDir,noise_file,fix_params,test=args.test,simtest=args.simtest)


if args.test:
    parlist = ['omch2','H0','As']
    parvals = [0.1194,67.0,2.2e-09]

    priorlist = ['H0']
    prioravg = np.array([67])
    priorwth = np.array([3])
    priorvals = np.array([prioravg,priorwth])

elif args.simtest:
    parlist = ['As']
    parvals = [2.2e-09]

    priorlist = []
    prioravg = np.array([])
    priorwth = np.array([])
    priorvals = np.array([prioravg,priorwth])


else:
    parlist = ['omch2','ombh2','H0','As','ns','massbias','yslope','scat']
    parvals = [0.1194,0.022,67.0,2.2e-09,0.96,0.80,0.08,0.2]
    #parvals = [  1.88435449e-01,   3.58611034e-02,   7.11553421e+01 ,  3.16378460e-09, 8.79223364e-01,   2.53233761e-02,   2.79267165e-02,   1.99945364e-01]
    #nan pars
   
    priorlist = ['ombh2','ns','H0','massbias','scat']
    prioravg = np.array([0.0223,0.96,67.3,0.68,0.2])
    priorwth = np.array([0.0009,0.02,3.6,0.11,0.1])
    prioravg = np.array([0.022,0.9624,67.3,0.68,0.2])
    priorwth = np.array([0.002,0.014,3.6,0.11,0.1])
    # prioravg = np.array([0.0223,0.96,67.3,0.8,0.2])
    # priorwth = np.array([0.0009,0.02,3.6,0.12,0.1])
    priorvals = np.array([prioravg,priorwth])

if (args.printtest):

    parvals2 = [3.46419819e-01,2.34697120e-02,6.50170056e+01,1.33398673e-09,9.36305025e-01,2.53310030e-01,1.93661978e-01,1.74839544e-01]
#parvals2 = [1.194e-01,2.34697120e-02,6.50170056e+01,1.33398673e-09,9.36305025e-01,2.53310030e-01,1.93661978e-01,1.74839544e-01]

    param_vals= CL.alter_fparams(fparams,parlist,parvals)
    cluster_props = np.array([CL.clst_z,CL.clst_zerr,CL.clst_y0*1e-4,CL.clst_y0err*1e-4])
    
    start = time.time()
    int_cc = ClusterCosmology(param_vals,CL.constDict,clTTFixFile=CL.clttfile) 
    
    print ('CC',time.time() - start)
    start = time.time()
    int_HMF = Halo_MF(int_cc,CL.mgrid,CL.zgrid)
    
    print ('HMF',time.time() - start)
    dn_dzdm_int = int_HMF.inter_dndmLogm(200.)

    zbins = 10
    LgYa = np.outer(np.ones(len(int_HMF.M.copy())),CL.LgY)
    Y = 10**LgYa
    Ma = np.outer(int_HMF.M.copy(),np.ones(len(LgYa[0,:])))
    clustind = 1


    print cluster_props[:,clustind]
    print parlist
    print parvals
    print "ln prob", np.log(CL.Prob_per_cluster(int_HMF,cluster_props[:,clustind],dn_dzdm_int,param_vals))
    print LgYa[-1,-1]
    Ytilde, theta0, Qfilt =simsTools.y0FromLogM500(np.log10(param_vals['massbias']*Ma/(param_vals['H0']/100.)), int_HMF.zarr[zbins], CL.tckQFit,sigma_int=param_vals['scat'],B0=param_vals['yslope'])
    print "ln Y val",np.log10(Y[-1,-1])
    print "ln Y~", np.log10(Ytilde[-1,-1])
    print Y[-1,30:35]/Ytilde[-1,-1]
    print np.log(Y[-1,-1]) - np.log(Ytilde[-1,-1])
    print "P of Y", CL.P_Yo(LgYa,int_HMF.M.copy(),int_HMF.zarr[zbins],param_vals)[-1,-1], 
 
    param_vals2= CL.alter_fparams(fparams,parlist,parvals2)
    int_cc2 = ClusterCosmology(param_vals2,CL.constDict,clTTFixFile=CL.clttfile) 
    int_HMF2 = Halo_MF(int_cc2,CL.mgrid,CL.zgrid)
    dn_dzdm_int2 = int_HMF2.inter_dndmLogm(200.)
    print
    print 'pars2', parvals2
    print "ln prop", np.log(CL.Prob_per_cluster(int_HMF2,cluster_props[:,clustind],dn_dzdm_int2,param_vals2))
    print LgYa[-1,-1]
    Ytilde, theta0, Qfilt =simsTools.y0FromLogM500(np.log10(param_vals2['massbias']*Ma/(param_vals2['H0']/100.)), int_HMF.zarr[zbins], CL.tckQFit,sigma_int=param_vals2['scat'],B0=param_vals2['yslope'])
    print "ln Y val",np.log10(Y[-1,-1])
    print "ln Y~", np.log10(Ytilde[-1,-1])
    #print np.log(Ytilde[-1,30:35])
    print Y[-1,-1]/Ytilde[-1,-1]
    print np.log(Y[-1,-1]/Ytilde[-1,-1])
    #print -1.*(np.log(Y[-1,30:35]/Ytilde[-1,30:35]))

    print "P of Y", CL.P_Yo(np.outer(np.ones(len(int_HMF.M.copy())),CL.LgY),int_HMF2.M.copy(),int_HMF2.zarr[zbins],param_vals2)[-1,-1]

    #pl = Plotter()
    #pl.add(np.log10(int_HMF.M200[:,zbins]),np.log10(int_HMF.dn_dM(int_HMF.M200,200)[:,zbins]*int_HMF.M200[:,zbins]),color='b',alpha=0.9)
    #pl.add(np.log10(int_HMF2.M200[:,zbins]),np.log10(int_HMF2.dn_dM(int_HMF2.M200,200)[:,zbins]*int_HMF2.M200[:,zbins]),color='r',alpha=0.9)
    #pl.add(np.log10(int_HMF.M200[:,zbins]),np.log10(dn_dzdm_int(int_HMF.zarr[zbins],np.log10(int_HMF.M.copy()))[:,0]*int_HMF.M200[:,zbins]),linestyle='--',color='g',alpha=0.9)
    #pl.add(np.log10(int_HMF2.M200[:,zbins]),np.log10(dn_dzdm_int2(int_HMF2.zarr[zbins],np.log10(int_HMF2.M.copy()))[:,0]*int_HMF2.M200[:,zbins]),linestyle='--',color='c',alpha=0.9)
    #pl.done("test_MF.png")

    #print np.log10(dn_dzdm_int(int_HMF.zarr[zbins],np.log10(int_HMF.M.copy()))[:,0]*int_HMF.M200[:,zbins])

    print "MF interp check", np.sum(np.log10(int_HMF.dn_dM(int_HMF.M200,200)[:,zbins]*int_HMF.M200[:,zbins]) - np.log10(dn_dzdm_int(int_HMF.zarr[zbins],np.log10(int_HMF.M.copy()))[:,0]*int_HMF.M200[:,zbins]))

    print int_HMF.zarr[zbins]
    
    start = time.time()
    print CL.lnlike(parvals,parlist)#,priorvals,priorlist)
    print ("like call", time.time() - start)
    print "prior",CL.lnprior(parvals,parlist,priorvals,priorlist)
#print "Prob",CL.lnprob(parvals,parlist,priorvals,priorlist)
    
    start = time.time()
    print CL.lnlike(parvals2,parlist)#,priorvals,priorlist)
    print ("like call", time.time() - start)
    print "prior",CL.lnprior(parvals2,parlist,priorvals,priorlist)
#print "Prob",CL.lnprob(parvals2,parlist,priorvals,priorlist)
    
    print parlist

    sys.exit(0)

Ndim, nwalkers = len(parvals), len(parvals)*2
P0 = np.array(parvals)

pos = [P0 + P0*1e-1*np.random.randn(Ndim) for i in range(nwalkers)]

start = time.time()

# pool = MPIPool()
# if not pool.is_master():
#     pool.wait()
#     sys.exit(0)

Nruns = args.nruns #int(1e6)
print (nwalkers,Nruns)
#nwalkers = 1
sampler = emcee.EnsembleSampler(nwalkers,Ndim,CL.lnprob,args =(parlist,priorvals,priorlist))#,pool=pool)
#sampler.run_mcmc(pos,Nruns)


filename = chain_out+"/sz_chain_"+args.chain_name+"_"+str(index)+".dat"
f = open(filename, "w")
f.close()

for result in sampler.sample(pos, iterations=Nruns, storechain=False):
    position = result[0]
    s8 = np.array(result[3]).reshape((len(result[3]),1))
    f = open(filename, "ab")
    savemat = np.concatenate((position,s8),axis=1)
    np.savetxt(f,savemat)
    f.close()
    print "Saved a sample."

# pool.close()

print (time.time() - start)  
