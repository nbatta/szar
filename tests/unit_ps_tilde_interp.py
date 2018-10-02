import numpy as np
from szar.counts import ClusterCosmology
from configparser import ConfigParser
from orphics.io import dict_from_section,list_from_config
from szar.clustering import Clustering
import sys

INIFILE = "input/pipeline.ini"
expName = 'S4-1.0-CDT'
gridName = 'grid-owl2'
version = '0.6'

clst = Clustering(INIFILE,expName,gridName,version)

mus = np.array([0])
ks = clst.HMF.kh
zs = clst.HMF.zarr

fine_zs = np.linspace(zs[0], zs[-1], 1000)

try:
    ps_interps = clst.ps_tilde_interpol(fine_zs, mus)
except Exception as e:
    print("Test failed at clustering.fine_sfunc")
    print(e)
    sys.exit()

expected = np.empty((ks.size, zs.size, mus.size))
if ps_interps != expected.shape:
    print("ps_tilde_interpol shape is not the expected value; test failed!")
    sys.exit()
else:
    print("Tests passed! (Check the plots though)")

coarse_ps_tils = clst.ps_tilde(mus)

plt.plot(zs, coarse_ps_tils[0,:,:], marker='o', label="coarse")
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel(r'$z_\ell$')
plt.ylabel(r'$\tilde P(z_\ell, k=k_{min})$')
plt.legend(loc='best')
plt.savefig('ps_tilde_interpols_test.svg')
