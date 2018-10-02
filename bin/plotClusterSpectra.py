import numpy as np
from szar.counts import ClusterCosmology
from configparser import ConfigParser
from orphics.io import dict_from_section,list_from_config
from szar.clustering import Clustering
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("inifile")
parser.add_argument("expname")
parser.add_argument("gridname")
parser.add_argument("version")
parser.add_argument("figname")
parser.add_argument("--legendloc", help="location of legend on figure", default="best")
args = parser.parse_args()

inifile = args.inifile
expName = args.expname
gridName = args.gridname
version = args.version
figname = args.figname
legendloc = args.legendloc 

#Do plotting - turn this into a function later
clst = Clustering(inifile,expName,gridName,version)

mus = np.array([0.5])
fsky = 1.
ks = clst.HMF.kh
delta_ks = np.diff(ks)


ps_bars = clst.ps_bar(mus, fsky)[:,0,0]
v_effs = clst.V_eff(mus, fsky, 1000)[:,0,0]
noise = np.sqrt(1/(v_effs[:-1] * (ks**2)[:-1] * delta_ks)) * ps_bars[:-1]

plt.plot(ks, ps_bars, label=r"$\bar P(k, \mu = 0.5)$")
plt.plot(ks[:-1], noise, label=r"Noise ($ \bar P/\sqrt{k^2 V_{eff} \Delta k}$)")
plt.plot(ks[:-1], ps_bars[:-1]/noise, label=r"$SNR$")
plt.xscale('log')
plt.yscale('log')

plt.xlabel(r'$k$')
#plt.ylabel(r'$ P_{lin}$')
plt.legend(loc=legendloc)

plt.savefig(figname)
