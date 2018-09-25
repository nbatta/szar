import numpy as np
from szar.counts import ClusterCosmology
from configparser import ConfigParser
from orphics.io import dict_from_section,list_from_config
from szar.clustering import Clustering
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

INIFILE = "input/pipeline.ini"

expName = 'S4-1.0-CDT'
gridName = 'grid-owl2'
version = '0.6'
clst = Clustering(INIFILE,expName,gridName,version)


n_ks = 200
ks = clst.HMF.kh
ZLO = 1e-3
ZUP = 1e-1

#ps

ps = clst.HMF.pk

plt.plot(ks, ps[0], label="$P_lin$")
plt.xscale('log')
plt.yscale('log')
#plt.xlabel(r'$k$')
#plt.ylabel(r'$P(k)$')
plt.xlim(ZLO, ZUP)
plt.legend(loc='lower left')

#ps_tilde
mus = np.array([0])
pstildes = clst.ps_tilde(mus)

plt.plot(ks, pstildes[0], label="$P_{tilde}$")
plt.xscale('log')
plt.yscale('log')
#plt.xlabel(r'$k$')
#plt.ylabel(r'$\tilde P(k, \mu=0)$')
plt.xlim(ZLO,ZUP)
plt.legend(loc='lower left')


#ps_bar
fsky = 1
ps_bar = clst.ps_bar(mus, fsky)

plt.plot(ks, ps_bar[0], '--', label="$P_{bar}$")
plt.xscale('log')
plt.yscale('log')

plt.xlabel(r'$k$')
plt.ylabel(r'$\bar P(k, \mu=0)$')
plt.xlim(ZLO, ZUP)
plt.legend(loc='lower left')

plt.savefig("ps_all_test.png")

plt.gcf().clear()

print(np.mean(ps[0]/ps_bar[0]))
