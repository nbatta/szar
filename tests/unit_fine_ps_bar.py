import numpy as np
from szar.counts import ClusterCosmology
from configparser import ConfigParser
from orphics.io import dict_from_section,list_from_config
from szar.clustering import Clustering
import sys
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

INIFILE = "input/pipeline.ini"
expName = 'S4-1.0-CDT'
gridName = 'grid-owl2'
version = '0.6'

clst = Clustering(INIFILE,expName,gridName,version)

mus = np.array([0])
ks = clst.HMF.kh
zs = clst.HMF.zarr
fsky = 1.

try:
    fine_sfunc_vals = clst.fine_sfunc(fsky, 1000)
except Exception as e:
    print("Test failed at clustering.fine_sfunc")
    print(e)
    sys.exit()

if fine_sfunc_vals.shape != zs.shape:
    print("fine_sfunc_vals shape is not the expected value; test failed!")
    sys.exit()
else:
    print("Tests passed! (Check the plots though)")

coarse_sfunc_vals = clst.Norm_Sfunc(fsky)

plt.plot(zs, 10*coarse_sfunc_vals, marker='o', label="coarse")
plt.plot(zs, 10*fine_sfunc_vals, marker='.', label="fine")
plt.plot(zs, coarse_sfunc_vals/fine_sfunc_vals, marker='.', label="ratio")
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel(r'$z_\ell$')
plt.ylabel(r'$10 \times S(z_\ell)$')
plt.legend(loc='upper center')
plt.savefig('fine_sfunc_test.svg')
