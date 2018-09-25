import numpy as np
from szar.counts import ClusterCosmology
from configparser import ConfigParser
from orphics.io import dict_from_section,list_from_config
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

INIFILE = "input/pipeline.ini"

Config = ConfigParser()
Config.optionxform=str
Config.read(INIFILE)

fparams = {}
for (key, val) in Config.items('params'):
    if ',' in val:
        param, step = val.split(',')
        fparams[key] = float(param)
    else:
        fparams[key] = float(val)

constDict = dict_from_section(Config,'constants')

clttfile = Config.get('general','clttfile')

cc = ClusterCosmology(fparams, constDict, clTTFixFile=clttfile)

Om = (cc.paramDict['omch2'] + cc.paramDict['ombh2'])/((cc.paramDict['H0']/100.)**2)

zarrs = np.linspace(0,9,1000)
scalefacs = 1/(1+zarrs)
scalefacs = np.flip(scalefacs)

gfunc = cc.growthfunc(zarrs)
gfunc_of_a = np.flip(gfunc)

plt.plot(scalefacs, gfunc_of_a)
plt.xscale('log')
plt.xlabel(r'$a$')
plt.ylabel(r'$D(a)$')
plt.savefig("growth_func_test.png")
plt.gcf().clear()

logderiv = cc.fgrowth(zarrs)

g = lambda x: (Om*(1 + x)**3/cc.E_z(x)**2)**0.55

plt.plot(zarrs, logderiv, label="Numerical")
plt.plot(zarrs, g(zarrs), label="$\Omega_m^\gamma$ estimate")
plt.xlabel(r'$z$')
plt.ylabel(r'$d\ln(D(a))/d\ln(a)$')
plt.legend(loc='lower right')
plt.savefig("logderiv_growth_test.png")

