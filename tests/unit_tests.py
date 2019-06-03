import numpy as np
from szar.counts import ClusterCosmology
from configparser import ConfigParser
from orphics.io import dict_from_section,list_from_config
from szar.clustering import Clustering
import sys
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import seaborn as sns

#Python 2 compatibility stuff
from six.moves import configparser
import six

if six.PY2:
  ConfigParser = configparser.SafeConfigParser
else:
  ConfigParser = configparser.ConfigParser
#

sns.set(style='whitegrid', font_scale=1.5, rc={"lines.linewidth": 2,'lines.markersize': 8.0,})

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

cc = ClusterCosmology(fparams,constDict,clTTFixFile=clttfile)

clst = Clustering(INIFILE,expName,gridName,version,cc)

def test_ps_tilde_interpol(cc, paper_plots=False):
    mus = np.linspace(0, 1, 3)
    ks = cc.HMF.kh
    zs = cc.HMF.zarr

    fine_zs = np.linspace(zs[0], zs[-1], 1000)
    ksamp_indices = np.linspace(0, ks.size, 4, dtype=int, endpoint=False)

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

    if paper_plots:
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        with sns.plotting_context("paper"):
            fig, ax = plt.subplots(mus.size, ksamp_indices.size, sharex='col')
            fig.set_figheight(6)
            fig.set_figwidth(10)
            for muindex,mu in enumerate(mus):
                for count,kindex in enumerate(ksamp_indices):
                    ax[muindex, count].plot(zs, coarse_ps_tils[kindex,:,muindex], marker='o', linestyle='None')
                    ax[muindex, count].plot(fine_zs, ps_interps[kindex,:,muindex])

            cols = [r'$k = {}$'.format(round(ks[kindex],3)) for kindex in ksamp_indices]
            rows = [r'$\mu = {}$'.format(round(mu, 3)) for mu in mus]

            for axis, col in zip(ax[0], cols):
                axis.set_title(col)

            for axis, row in zip(ax[:,0], rows):
                axis.set_ylabel(row, size='large')

            for axis in ax[-1]:
                axis.set_xlabel(r'$z$')

            fig.tight_layout()
            fig.savefig('ps_tilde_interpols_test.pdf')
    else:
        plt.plot(zs, coarse_ps_tils[0,:,:], marker='o', label="on grid")
        plt.plot(fine_zs, ps_interps[0,:,:], label="interpolated")
        plt.xlabel(r'$z_\ell$')
        plt.ylabel(r'$\tilde P(z_\ell, k={})$'.format(ks[0]))
        plt.legend(loc='best')
        plt.savefig('ps_tilde_interpols_test.pdf')

    plt.gcf().clear()

def test_fine_ps_bar(cc, nsamps):
    mus = np.array([0, 0.75, 0.95, 1])
    ks = cc.HMF.kh
    zs = cc.HMF.zarr

    try:
        fine_ps_bars = cc.fine_ps_bar(mus, nsamps)
    except Exception as e:
        print("Test of fine_ps_bar failed at clustering.fine_ps_bars")
        print(e)
        return

    expected = np.empty((ks.size, zs.size - 2, mus.size))
    if fine_ps_bars.shape != expected.shape:
        print("fine_ps_bar shape is not the expected value; test failed!")
        return
    else:
        print("Tests of fine_ps_bar passed! (Check the plots though)")

    coarse_ps_bar = cc.ps_bar(mus)

    def _ps_bar_integrand(finer_zs, mus):
        dvdz = cc.dVdz_fine(finer_zs)
        ntils = cc.ntilde_interpol(finer_zs)
        ps_tils = cc.ps_tilde_interpol(finer_zs, mus)
        prefac = dvdz * ntils**2
        prefac = prefac[..., np.newaxis]
        return prefac * ps_tils

    plt.plot(zs, coarse_ps_bar[0,:,:], marker='o', label="coarse")
    plt.plot(zs[1:-1], fine_ps_bars[0,:,:], marker='.', label="fine")
    plt.xlabel(r'$z_\ell$')
    plt.ylabel(r'$\bar P(z_\ell, \mu = 0, k={})$'.format(ks[0]))
    plt.legend(loc='best')
    plt.savefig('fine_ps_bars_test_nsamps{}.pdf'.format(nsamps))

    plt.gcf().clear()

    plt.plot(ks, coarse_ps_bar[:,18,:], label="coarse")
    plt.plot(ks, fine_ps_bars[:,17,:], label="fine")
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim((1e-1, 1e8))
    plt.xlabel(r'$k$')
    plt.ylabel(r'$\bar P(z = {}, \mu = 0, k)$'.format(round(zs[1],3)))
    plt.legend(loc='best')
    plt.savefig('fine_ps_bars_kspace_nsamps{}.pdf'.format(nsamps))

    plt.gcf().clear()

    finer_zs = np.linspace(zs[1], zs[-1], 10*nsamps)
    integrand = _ps_bar_integrand(finer_zs, mus)

    plt.plot(finer_zs, integrand[0, :, 0])
    plt.xlabel(r'$z$')
    plt.ylabel(r'$\frac{dV}{dz} \, \tilde n^2\, \tilde P$')
    plt.savefig('fine_ps_bar_integrand_test_nsamps{}.pdf'.format(nsamps))

    plt.gcf().clear()

def test_fine_sfunc(cc):
    mus = np.linspace(0, 1, 3)
    ks = cc.HMF.kh
    zs = cc.HMF.zarr
    zs_noends = zs[1:-1]
    scale = 100
    nsubsamples = 100

    try:
        fine_sfunc_vals = cc.fine_sfunc(nsubsamples)
    except Exception as e:
        print("Test of fine_sfunc failed at clustering.fine_sfunc")
        print(e)
        sys.exit()

    expected = zs_noends
    if fine_sfunc_vals.shape != expected.shape:
        print("fine_sfunc_vals shape is not the expected value; test failed!")
        sys.exit()
    else:
        print("Tests of fine_sfunc passed! (Check the plots though)")

    coarse_sfunc_vals = cc.Norm_Sfunc()

    plt.plot(zs, scale*coarse_sfunc_vals, marker='o', label=r"$N = 1$")
    plt.plot(zs_noends, scale*fine_sfunc_vals, marker='.', label=r"$N={}$".format(nsubsamples), linestyle='dashed')
    plt.plot(zs_noends, coarse_sfunc_vals[1:-1]/fine_sfunc_vals, marker='.', label="ratio", linestyle='dashdot')
    #plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel(r'$z_\ell$')
    plt.ylabel(r'${} \times S(z_\ell)$'.format(scale))
    plt.legend(loc='best')
    plt.savefig('fine_sfunc_test.pdf')

    plt.gcf().clear()

def test_ps_tilde(cc):
    mus = np.linspace(0,1,3)
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

def test_ntils(cc):
    mus = np.linspace(0, 1, 3)
    ks = cc.HMF.kh
    zs = cc.HMF.zarr

    fine_zs = np.linspace(zs[0], zs[-1], 1000)
    ksamp_indices = np.linspace(0, ks.size, 4, dtype=int, endpoint=False)

    ntils = cc.ntilde()
    ntils_interp = cc.ntilde_interpol(fine_zs)

    plt.plot(zs, ntils, marker='o', label="on grid")
    plt.plot(fine_zs, ntils_interp, label="interpolated")
    plt.xlabel(r'$z_\ell$')
    plt.ylabel(r'$\tilde n(z_\ell)$')
    plt.legend(loc='best')
    plt.savefig('n_tilde_interpols_test.pdf')
    plt.gcf().clear()


if __name__ == '__main__':
    test_fine_sfunc(clst)
    test_ps_tilde(clst)
    test_ps_tilde_interpol(clst)
    test_ntils(clst)
    nsamps = 100
    test_fine_ps_bar(clst, nsamps)
