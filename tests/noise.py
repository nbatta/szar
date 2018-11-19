import numpy as np
from configparser import ConfigParser
from orphics.io import dict_from_section, list_from_config
from szar.counts import ClusterCosmology
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

Config = ConfigParser()
Config.optionxform=str
Config.read(INIFILE)
clttfile = Config.get('general','clttfile')
constDict = dict_from_section(Config,'constants')

fparams = {}
for (key, val) in Config.items('params'):
    if ',' in val:
        param, step = val.split(',')
        fparams[key] = float(param)
    else:
        fparams[key] = float(val)

expName = 'S4-1.0-CDT'
gridName = 'grid-owl2'
version = '0.6'
cc = ClusterCosmology(fparams,constDict,clTTFixFile=clttfile)
clst = Clustering(INIFILE,expName,gridName,version,cc)

fsky = 1.

ks = clst.HMF.kh
delta_ks = np.gradient(ks)
zs = clst.HMF.zarr[1:-1]
mus = np.linspace(-1, 1, 9)
deltamu = np.gradient(mus)

ps_bars = clst.fine_ps_bar(mus)

v_effs = clst.V_eff(mus)

noise = np.sqrt(8) * np.pi * np.sqrt(1/(deltamu * v_effs * (ks**2)[..., np.newaxis, np.newaxis] * delta_ks[..., np.newaxis, np.newaxis])) * ps_bars

snr = ps_bars/noise

print(noise.shape)
print(np.einsum('ijk,ijk', noise, noise))

print(np.sum(snr**2))
