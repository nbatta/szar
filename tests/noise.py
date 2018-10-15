import numpy as np
from configparser import ConfigParser
from orphics.io import dict_from_section, list_from_config
from szar.clustering import Clustering
from scipy.integrate import simps
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import sys

def _test_veff(cc, mu, fsky, nsubsamples):
    V0 = cc.v0(fsky, nsubsamples)
    V0 = np.reshape(V0, (V0.size,1))

    nbar = cc.ntilde()[1:-1]
    nbar = np.reshape(nbar, (nbar.size,1))

    ps = cc.fine_ps_bar(mu,fsky, nsubsamples)
    ps /= 4 * np.pi #Replicates earlier errant 4 pi factors

    npfact = np.multiply(nbar, ps)
    frac = npfact/(1. + npfact)

    ans = np.multiply(frac**2,V0)
    return ans

INIFILE = "input/pipeline.ini"

expName = 'S4-1.0-CDT'
gridName = 'grid-owl2'
version = '0.6'
clst = Clustering(INIFILE,expName,gridName,version)

fsky = 1.

ks = clst.HMF.kh
delta_ks = np.diff(ks)
zs = clst.HMF.zarr[1:-1]
mus = np.array([0])

ps_bars = clst.fine_ps_bar(mus, fsky, 100)[:,0,0]
test_ps_bars = ps_bars/(4 * np.pi) #Again replicates errant 4 pis

v_effs = clst.V_eff(mus, fsky, 100)[:,0,0]
test_veffs = _test_veff(clst, mus, fsky, 100)[:,0,0]

noise = np.sqrt(8) * np.pi * np.sqrt(1/(v_effs[:-1] * (ks**2)[:-1] * delta_ks)) * ps_bars[:-1]
test_noise = np.sqrt(1/(test_veffs[:-1] * (ks**2)[:-1] * delta_ks)) * test_ps_bars[:-1] # Noise with the incorrect factors of 4 pi

plt.plot(ks, ps_bars, label=r"$\bar P(k, \mu = 0)$")
plt.plot(ks[:-1], noise, label=r"noise")
plt.plot(ks[:-1], ps_bars[:-1]/noise, label=r"$SNR$")
plt.plot(ks[:-1], test_noise, label=r"$4\pi$ scaled noise")
plt.xscale('log')
plt.yscale('log')

plt.xlabel(r'$k$')
#plt.ylabel(r'$ P_{lin}$')
plt.legend(loc='best')

plt.savefig('noise_test.svg')
