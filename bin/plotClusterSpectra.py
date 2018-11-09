import numpy as np
from szar.counts import ClusterCosmology
from configparser import ConfigParser
from orphics.io import dict_from_section,list_from_config
from szar.clustering import Clustering
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import seaborn as sns
sns.set(style='whitegrid', font_scale=1.5, rc={"lines.linewidth": 2,'lines.markersize': 8.0,})
import argparse
#Python 2 compatibility stuff
from six.moves import configparser
import six

if six.PY2:
  ConfigParser = configparser.SafeConfigParser
else:
  ConfigParser = configparser.ConfigParser

def get_cc(ini):
    Config = ConfigParser()
    Config.optionxform=str
    Config.read(ini)
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
    return cc

def make_plots_fid(clst, figname, legendlog):
    mus = np.linspace(-1,1,9)
    fsky = 1.
    ks = clst.HMF.kh
    delta_ks = np.diff(ks)
    nsamples = 1000

    ps_bars = clst.fine_ps_bar(mus)[:,0,0]
    v_effs = clst.V_eff(mus)[:,0,0]
    noise = np.sqrt(8) * np.pi * np.sqrt(1/(v_effs[:-1] * (ks**2)[:-1] * delta_ks)) * ps_bars[:-1]

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    with sns.plotting_context("paper"):
        plt.plot(ks, ps_bars, label=r"$\bar P(k, \mu = 0)$")
        plt.plot(ks[:-1], noise, label=r"Noise ($ \bar P/\sqrt{k^2 V_{eff} \Delta k}$)")
        plt.plot(ks[:-1], ps_bars[:-1]/noise, label=r"$SNR$")
        plt.xscale('log')
        plt.yscale('log')

        plt.xlabel(r'$k$')
        #plt.ylabel(r'$ P_{lin}$')

        plt.legend(loc=legendloc)

        plt.savefig(figname)
        plt.gcf().clear()

def make_plots_upvdown(clst, ups, downs, factors, params, figname, dir_, legendloc):
    mus = np.linspace(-1,1,9)
    ks = clst.HMF.kh
    zs = clst.HMF.zarr[1:-1]
    delta_ks = np.diff(ks)

    param_index = {key:index for index,key in enumerate(params.keys())}
    
    ps_bars_fid = clst.fine_ps_bar(mus)
    noise = 1/np.sqrt(factors)

    def _plot_ps_diff(param, index):
        plt.plot(ks, ps_bars_fid[:,0,0], label=r"fid")
        plt.plot(ks, ups[index][:,0,0] - ps_bars_fid[:,0,0], label=r"up - fid")
        plt.plot(ks, downs[index][:,0,0] - ps_bars_fid[:,0,0], label=r"down - fid")
        plt.plot(ks, ps_bars_fid[:,0,0]/noise[:,0,0], label="SNR")
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(loc=legendloc)
        plt.title(f'param: ${param}$')

        plt.savefig(dir_ + f'{param}_diff.svg')
        plt.gcf().clear()

    def _plot_ps(param, index):
        plt.plot(ks, ps_bars_fid[:,0,0], label=r"fid")
        #plt.plot(ks, ups[index][:,0,0], label=r"up")
        plt.plot(ks, downs[index][:,0,0], label=r"down", linestyle=":")
        plt.plot(ks, ps_bars_fid[:,0,0]/noise[:,0,0], label="SNR")
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(loc=legendloc)
        plt.title(f'param: ${param}$')

        plt.savefig(dir_ + f'{param}_updown.svg')
        plt.gcf().clear()

    def _plot_table_ps(param, index):
        zsamp_indices = np.linspace(0, zs.size, 4, dtype=int, endpoint=False)
        musamp_indices = np.linspace(0, mus.size, 2, dtype=int, endpoint=False)

        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        #with sns.plotting_context("paper"):
        fig, ax = plt.subplots(musamp_indices.size, zsamp_indices.size, sharex='col')
        fig.set_figheight(6)
        fig.set_figwidth(12)    
        for mucount,muindex in enumerate(musamp_indices):
            for zcount,zindex in enumerate(zsamp_indices):
                ax[mucount, zcount].plot(ks, ps_bars_fid[:, zindex, muindex], label=r"fid")
                ax[mucount, zcount].plot(ks, ups[index][:, zindex, muindex] - ps_bars_fid[:, zindex, muindex], label=f"${param}$ up diff")
                ax[mucount, zcount].plot(ks, downs[index][:, zindex, muindex] - ps_bars_fid[:, zindex, muindex], label=f"${param}$ down diff")
                ax[mucount, zcount].plot(ks, ps_bars_fid[:, zindex, muindex]/noise[:, zindex, muindex], label="SNR")
                ax[mucount, zcount].set_yscale('log')
                ax[mucount, zcount].set_xscale('log')

        cols = [r'$z = {}$'.format(round(zs[zindex],3)) for zindex in zsamp_indices]
        rows = [r'$\mu = {}$'.format(round(mus[muindex], 3)) for muindex in musamp_indices]

        for axis, col in zip(ax[0], cols):
            axis.set_title(col)

        for axis, row in zip(ax[:,0], rows):
            axis.set_ylabel(row, size='large')

        #for axis in ax[-1]:
        #    axis.set_xlabel(r'$k$')

        fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.12)  # create some space below the plots by increasing the bottom-value
        ax.flatten()[-2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4)
        fig.tight_layout()
        fig.savefig(dir_ + f'{param}_diff_table.svg')

    #for param in params.keys():
    _plot_ps('mnu', param_index['mnu'])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inifile", help="location of inifile", default="input/pipeline.ini")
    parser.add_argument("-e", "--expname", help="name of experiment", default='S4-1.0-CDT')
    parser.add_argument("-g", "--gridname", help="name of grid", default='grid-owl2')
    parser.add_argument("-v", "--version", help="version number", default='0.6')
    parser.add_argument("figname", help="desired figure output filename")
    parser.add_argument("--legendloc", help="location of legend on figure", default="best")
    parser.add_argument("-u", "--upfile", help="up-varied power spectra file")
    parser.add_argument("-d", "--downfile", help="down-varied power spectra file")
    parser.add_argument("-p", "--prefacfile", help="fisher prefactor file")
    parser.add_argument("-par", "--params", help="fisher parameters")
    args = parser.parse_args()

    DIR = 'figs/'

    params = np.load(args.params).item()
    psups = np.load(args.upfile)
    psdowns = np.load(args.downfile)
    prefacs = np.load(args.prefacfile)
    
    cc = get_cc(args.inifile)
    clst = Clustering(args.inifile, args.expname, args.gridname, args.version, cc)
    make_plots_upvdown(clst, psups, psdowns, prefacs, params, args.figname, DIR, args.legendloc)


if __name__ == '__main__':
    main()
