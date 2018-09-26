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

FSKY = 1.

coarse_v0_vals = clst.v0(FSKY)
zs = clst.HMF.zarr
zedges = clst.HMF.zarr_edges

ntildes = clst.ntilde()

def fine_v0(fsky, zgrid, zgridedges, npoints):
    values = np.zeros(zgrid.size)
    for i in range(zgrid.size):
        fine_zs = np.linspace(zgridedges[i], zgridedges[i+1], npoints)
        dvdz = clst.dVdz_fine(fine_zs)
        integral = simps(dvdz, fine_zs)
        values[i] = integral * 4 * np.pi * fsky
    return values

import time
start = time.time()
fine_v0_vals = fine_v0(FSKY, zs, zedges, 100)
end = time.time()
print(end - start)

fine_zs = np.linspace(zs[0], zs[-1], 1000)

#plt.plot(zs, coarse_v0_vals, label="coarse", marker='o')
#plt.plot(zs, fine_v0_vals, '--', label="fine", marker='.')
#plt.plot(fine_zs, clst.dVdz_fine(fine_zs), label="dVdz")
#plt.plot(zs, coarse_v0_vals/fine_v0_vals, label="ratio")
plt.plot(zs, ntildes**2 * clst.HMF.dVdz, label="dV/dz * n^2")
#plt.xscale('log')
#plt.yscale('log')
#for z in zs:
#    plt.axvline(x=z)
plt.xlabel(r'$z_\ell$')
plt.ylabel(r'$V_0(z_\ell)$')
plt.legend(loc='lower left')
plt.savefig('nsqr_dvdz_test.png')
