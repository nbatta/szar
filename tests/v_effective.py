import numpy as np
from configparser import ConfigParser
from orphics.io import dict_from_section, list_from_config
from szar.clustering import Clustering
from scipy.integrate import simps
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import sys

def w_v(cc, mu, fsky):
    nbar = cc.ntilde()
    nbar = np.reshape(nbar, (nbar.size,1))

    ps = cc.ps_bar(mu,fsky)
    npfact = np.multiply(nbar, ps)
    frac = npfact/(1. + npfact)
    return frac

INIFILE = "input/pipeline.ini"

expName = 'S4-1.0-CDT'
gridName = 'grid-owl2'
version = '0.6'
clst = Clustering(INIFILE,expName,gridName,version)

fsky = 1.

ks = clst.HMF.kh
zs = clst.HMF.zarr[1:-1]
mus = np.array([0])

v0s = clst.v0(fsky, 1000)

v_effs = clst.V_eff(mus, fsky, 1000)

plt.plot(zs, v0s, label=r"$V_0$")
plt.plot(zs, v_effs[43,:,:], label=r"$V_{eff}(k\approx 0.001)$")
plt.plot(zs, v_effs[114,:,:], label=r"$V_{eff}(k\approx 0.05)$")
plt.plot(zs, v_effs[127,:,:], label=r"$V_{eff}(k\approx 0.1)$")
plt.plot(zs, v_effs[139,:,:], label=r"$V_{eff}(k\approx 0.2)$")
#plt.plot(ks, v_effs[0])
#plt.xscale('log')
plt.xlim((0, 1.8))
plt.yscale('log')
plt.xlabel(r'$z$')
plt.ylabel(r'$V_{eff}(k)$')
plt.legend(loc='best')
plt.savefig('volumes_test.svg')

plt.gcf().clear()

wvs = w_v(clst, mus, fsky)

plt.plot(ks, wvs[:,4,:], label=r"$W(z = 0.45)$")
plt.plot(ks, wvs[:,9,:], label=r"$W(z = 0.95)$")
plt.plot(ks, wvs[:,14,:], label=r"$W(z = 1.45)$")
plt.xlim((1e-3, 5e-1))
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$k$')
plt.ylabel(r'$W_v(k)$')
plt.legend(loc='best')
plt.savefig('wv_test.svg')
