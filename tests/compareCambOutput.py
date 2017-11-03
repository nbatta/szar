"""
This script compares the output of CAMB_wa/out.ini with the
output of pycamb for our fiducial cosmology with wa=0. This
instills confidence in our ability to directly read in 
the matter power spectrum output by CAMB_wa/out.ini for
wa!=0 cosmologies when generating mass functions for 
Fisher matrices.

Feb 24, 2017 - MM 
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
from szar.counts import ClusterCosmology,SZ_Cluster_Model,Halo_MF,SampleVariance
from orphics.tools.io import Plotter,dictFromSection,listFromConfig
from configparser import SafeConfigParser 
from orphics.tools.io import Plotter
from scipy.interpolate import interp1d

clusterParams = 'LACluster' # from ini file
cosmologyName = 'LACosmology' # from ini file
iniFile = "input/params.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)

lmax = 3000
cosmoDict = dictFromSection(Config,cosmologyName)
constDict = dictFromSection(Config,'constants')
clusterDict = dictFromSection(Config,clusterParams)
cc = ClusterCosmology(cosmoDict,constDict,lmax,pickling=True)
HMF = Halo_MF(cc)

cambOutFile = lambda i: "/home/msyriac/software/CAMB_wa/testWaFid_matterpower_"+str(i)+".dat"

zrange = np.arange(0.,3.05,0.05)

kh, z, pk, s8 = HMF.pk(zrange)

#pl = Plotter(scaleY='log',scaleX='log')
pl = Plotter()

Nzs = pk.shape[0]

for i,z in enumerate(zrange[::-1]):

    kh_camb,P_camb = np.loadtxt(cambOutFile(i),unpack=True)

    if i==0:
        kmin = max(kh_camb[0],kh[0])
        kmax = min(kh_camb[-1],kh[-1])
        keval = np.logspace(np.log10(kmin),np.log10(kmax),20)

    pcambfunc = interp1d(kh_camb,P_camb)
    pfunc = interp1d(kh,pk[Nzs-i-1,:])

    pcambeval = pcambfunc(keval)
    peval = pfunc(keval)
    pdiff = (pcambeval-peval)*100./peval
    print((z,pdiff))

    if i%1==0:
        pl.add(keval,pdiff)
        #pl.add(kh_camb,P_camb)
        #pl.add(kh,pk[Nzs-i-1,:],ls="--")

pl.done("output/testcamb.png")
