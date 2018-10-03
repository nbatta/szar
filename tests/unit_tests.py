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

def test_ps_tilde_interpol(cc):
    mus = np.array([0])
    ks = cc.HMF.kh
    zs = cc.HMF.zarr

    fine_zs = np.linspace(zs[0], zs[-1], 1000)

    try:
        ps_interps = cc.ps_tilde_interpol(fine_zs, mus)
    except Exception as e:
        print("Test of ps_tilde_interpol failed at clustering.tilde_interpol")
        print(e)
        return

    expected = np.empty((ks.size, fine_zs.size, mus.size))
    if ps_interps.shape != expected.shape:
        print("ps_tilde_interpol shape is not the expected value; test failed!")
        sys.exit()
    else:
        print("Tests of ps_tilde_interpol passed! (Check the plots though)")

    coarse_ps_tils = cc.ps_tilde(mus)

    plt.plot(zs, coarse_ps_tils[0,:,:], marker='o', label="coarse")
    plt.plot(fine_zs, ps_interps[0,:,:], label="interp\'d")
    #plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel(r'$z_\ell$')
    plt.ylabel(r'$\tilde P(z_\ell, k=k_{min})$')
    plt.legend(loc='best')
    plt.savefig('ps_tilde_interpols_test.svg')

    plt.gcf().clear()

#WIP
def test_fine_ps_bar(cc):
    mus = np.array([0])
    ks = cc.HMF.kh
    zs = cc.HMF.zarr
    fsky = 1.

    #try:
    fine_ps_bars = cc.fine_ps_bar(mus, fsky, 100)
    #except Exception as e:
    #   print("Test of fine_ps_bar failed at clustering.fine_ps_bars")
    #   print(e)
    return

    expected = np.empty((ks.size, 100, mus.size))
    print(expected.shape)
    print(fine_ps_bars.shape)
    if fine_ps_bars.shape != expected.shape:
        print("fine_ps_bar shape is not the expected value; test failed!")
        return
    else:
        print("Tests of fine_ps_bar passed! (Check the plots though)")

    coarse_psbar_vals = cc.ps_bar(mus, fsky)

    plt.plot(zs, coarse_ps_bar, marker='o', label="coarse")
    plt.plot(zs, fine_ps_bars, marker='.', label="fine")
    plt.xlabel(r'$z_\ell$')
    plt.ylabel(r'$\bar P(z_\ell, \mu = 0, k=m_{min})$')
    plt.legend(loc='upper center')
    plt.savefig('fine_ps_bars_test.svg')

    plt.gcf().clear()

def test_fine_sfunc(cc):
    mus = np.array([0])
    ks = cc.HMF.kh
    zs = cc.HMF.zarr
    fsky = 1.

    try:
        fine_sfunc_vals = cc.fine_sfunc(fsky, 1000)
    except Exception as e:
        print("Test of fine_sfunc failed at clustering.fine_sfunc")
        print(e)
        sys.exit()

    if fine_sfunc_vals.shape != zs.shape:
        print("fine_sfunc_vals shape is not the expected value; test failed!")
        sys.exit()
    else:
        print("Tests of fine_sfunc passed! (Check the plots though)")

    coarse_sfunc_vals = cc.Norm_Sfunc(fsky)

    plt.plot(zs, 10*coarse_sfunc_vals, marker='o', label="coarse")
    plt.plot(zs, 10*fine_sfunc_vals, marker='.', label="fine")
    plt.plot(zs, coarse_sfunc_vals/fine_sfunc_vals, marker='.', label="ratio")
    #plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel(r'$z_\ell$')
    plt.ylabel(r'$10 \times S(z_\ell)$')
    plt.legend(loc='upper center')
    plt.savefig('fine_sfunc_test.png')

    plt.gcf().clear()

def test_ps_tilde(cc):
    mus = np.linspace(0,1, 50)
    ks = cc.HMF.kh
    zs = cc.HMF.zarr
    try:
        pstildes = cc.ps_tilde(mus)
    except Exception as e:
        print("Test of ps_tilde failed at clustering.ps_tilde")
        print(e)
        return

    expected = np.empty((ks.size, zs.size, mus.size))

    if pstildes.shape != expected.shape:
        print("ps_tilde shape is not the expected value; test failed!")
        return
    else:
        print("Tests of ps_tilde passed!")

if __name__ == '__main__':
    #test_fine_sfunc(clst)
    #test_ps_tilde(clst)
    #test_ps_tilde_interpol(clst)
    test_fine_ps_bar(clst)
