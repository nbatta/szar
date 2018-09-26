import numpy as np
from configparser import ConfigParser
from orphics.io import dict_from_section, list_from_config
from szar.clustering import Clustering
from scipy.integrate import simps
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

INIFILE = "input/pipeline.ini"

expName = 'S4-1.0-CDT'
gridName = 'grid-owl2'
version = '0.6'
clst = Clustering(INIFILE,expName,gridName,version)

fsky = 1.

ks = clst.HMF.kh
zs = clst.HMF.zarr
mus = np.array([0])

#v0s = clst.v0(fsky)
v_effs = clst.V_eff(mus, fsky)

#plt.plot(zs, v0s, label="$V_0$")
plt.plot(ks, v_effs[0])
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$k$')
plt.ylabel(r'$V_{eff}(k)$')
plt.legend(loc='lower right')
plt.savefig('veff_test.svg')
