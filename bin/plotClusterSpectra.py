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

mus = np.array([0])
fsky = 1.
ks = clst.HMF.kh

ps_bars = clst.ps_bar(mus, fsky)[0]
v_effs = clst.V_eff(mus, fsky)[0]
noise = np.sqrt(1/v_effs)/ks

plt.plot(ks, ps_bars, label=r"$\bar P(k, \mu = 0)$")
plt.plot(ks, noise, label="Noise ($1/\sqrt{k^2 V_{eff}}$)")
plt.plot(ks, ps_bars/noise, label=r"$SNR$")
plt.xscale('log')
plt.yscale('log')

plt.xlabel(r'$k$')
#plt.ylabel(r'$ P_{lin}$')
plt.legend(loc=legendloc)

plt.savefig(figname)
