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

def _get_latex_params(inifile):
    config = ConfigParser()
    config.optionxform=str
    config.read(inifile)

    latex_param_list = config.items('fisher-clustering', 'paramLatexList')[1][1].split(',')
    return latex_param_list

def make_plots_upvdown(ini, clst, ups, downs, factors, params, figname, dir_, legendloc):
    mus = np.linspace(-1,1,9)
    ks = clst.HMF.kh
    kinds = np.where(ks <= 0.14)[0]
    ks = ks[ks <= 0.14]
    zs = clst.HMF.zarr[1:-1]
    delta_ks = np.gradient(ks)
    delta_mus = np.gradient(mus)

    ups = ups[:, kinds, ...]
    downs = downs[:, kinds, ...]

    param_index = {key:index for index,key in enumerate(params.keys())}

    latex_params = _get_latex_params(ini)
    latex_paramdict = {}
    for index,key in enumerate(params):
        latex_paramdict[key] = latex_params[index]

    ps_bars_fid = clst.fine_ps_bar(mus)[kinds, ...]

    noise = 1/np.sqrt(factors[kinds, ...])

    v0 = clst.v0(100)
    ntil = clst.ntilde()[1:-1]
    flat_noise = 1/ntil
    shot_shape = 1/(ks[..., np.newaxis]*ntil[np.newaxis, ...])
    sn_prefac = np.sqrt((8 * np.pi**2)/(delta_ks[..., np.newaxis, np.newaxis] * delta_mus[np.newaxis, np.newaxis, ...] * v0[np.newaxis, ..., np.newaxis]))
    sn_prefac_permode = np.sqrt((8 * np.pi**2)/(v0[np.newaxis, ..., np.newaxis]))
    shot_noise = 1/(ks[..., np.newaxis, np.newaxis] * ntil[np.newaxis, ..., np.newaxis]) * sn_prefac
    shot_noise_permode = 1/(ks[..., np.newaxis, np.newaxis] * ntil[np.newaxis, ..., np.newaxis]) * sn_prefac_permode
    cosmic_var = ps_bars_fid/(ks[..., np.newaxis, np.newaxis]) * sn_prefac
    cosmic_var_permode = ps_bars_fid/(ks[..., np.newaxis, np.newaxis]) * sn_prefac_permode

    def _plot_ps_diff(param, index):
        #plt.plot(ks, ps_bars_fid[:,0,0], label=r"fid")
        plt.plot(ks, ups[index][:,0,0] - ps_bars_fid[:,0,0], label=r"up - fid")
        plt.plot(ks, downs[index][:,0,0] - ps_bars_fid[:,0,0], label=r"down - fid", linestyle=':')
        #plt.plot(ks, ps_bars_fid[:,0,0]/noise[:,0,0], label="SNR")
        plt.xscale('symlog')
        #plt.yscale('symlog')
        plt.legend(loc=legendloc)
        plt.title(f'param: ${param}$')

        plt.savefig(dir_ + figname + '_' + f'{param}_diff.svg')
        plt.gcf().clear()

    def _plot_ps(param, index, zindex, muindex):
        fid = ps_bars_fid[:, zindex, muindex]
        up = ups[index][:, zindex, muindex]
        down = downs[index][:, zindex, muindex]

        #plt.plot(ks, fid, zindex, muindex], label=r"$P(p_\alpha)$")
        plt.plot(ks, up/fid, label=r"$\bar P(p_\alpha + \epsilon_\alpha)$", linestyle='--')
        plt.plot(ks, down/fid, label=r"$\bar P(p_\alpha - \epsilon_\alpha)$", linestyle=":")
        #plt.plot(ks, ps_bars_fid[:, zindex, muindex]/noise[:, zindex, muindex], label=r"$\bar P(p_\alpha)/\mathrm{noise}$")
        #plt.xscale('log')
        #plt.yscale('log')
        plt.legend(loc=legendloc)
        #plt.title(f'param: ${param}$')

        plt.savefig(dir_ + figname + '_' + f'{param}_updown.svg')
        plt.gcf().clear()

    def _plot_ps_with_ratio(param, index, zindex, muindex):
        fid = ps_bars_fid[:, zindex, muindex]
        up = ups[index][:, zindex, muindex]
        down = downs[index][:, zindex, muindex]

        nse = noise[:, zindex, muindex]
        flat_nse = np.array([flat_noise[zindex] for i in range(len(ks))])
        shot_nse = shot_noise[:, zindex, muindex]
        cosm_var = cosmic_var[:, zindex, muindex]
        shot_nse_permode = shot_noise[:, zindex, muindex]
        cosm_var_permode = cosmic_var[:, zindex, muindex]
        snr = ps_bars_fid[:, zindex, muindex]/nse

        latexp = latex_paramdict[param]
        z = zs[zindex]
        musqr = mus[muindex]**2

        k_snr = ks[np.where( np.abs(snr - 1) < 0.1)]

        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"

        fig, ax = plt.subplots(2, sharex=True)

        ax[0].plot(ks, fid, label=r"$\bar P$")
        #ax[0].fill_between(ks, fid - nse, fid + nse,
        #         color='grey', alpha=0.2)
        #ax[0].plot(ks, nse, label=r"$\mathrm{{total\, noise}}$", linestyle='--')
        ax[0].plot(ks, flat_nse, label=r"$1/\tilde n$", linestyle="--")
        ax[0].plot(ks, flat_nse + fid, label=r'$\bar P + 1/\tilde n$', linestyle=':')
        #ax[0].plot(ks, shot_nse_permode, label=r"$\mathrm{shot\, noise/mode}$", linestyle=':')
        #ax[0].plot(ks, cosm_var_permode, label=r"$\mathrm{cosmic\, var/mode}$", linestyle='-.')
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        ax[0].legend(loc='best')
        #ax[0].axvspan(k_snr[0], k_snr[-1], alpha=0.1, color='blue')
        ax[0].set_ylabel(r'$\bar P(k)$')
        ax[0].set_title(f'$z = {round(z,3)}, \quad \mu^2 = {round(musqr,3)}$')
        #ax[0].set_aspect(1)

        ax[1].plot(ks, up/fid, label=r"$\bar P({0} + \epsilon)/\bar P({0})$".format(latexp), color=sns.xkcd_palette(['green'])[0])
        ax[1].plot(ks, down/fid, label=r"$\bar P({0} - \epsilon)/\bar P({0})$".format(latexp), linestyle='--', color=sns.xkcd_palette(['pinkish red'])[0])
        #ax[1].axvspan(k_snr[0], k_snr[-1], alpha=0.1, color='blue')
        ax[1].set_xlabel(r'$k$')
        ax[1].set_ylabel(r'$\Delta \bar P(k)$')
        #ax[2].set_yscale('log')
        ax[1].set_xscale('log')
        ax[1].legend(loc='center left')
        #ax[1].legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2)
        #ax[1].set_aspect(1)

        fig.tight_layout()
        fig.set_size_inches(5.8,9)
        fig.savefig(dir_ + figname + '_' + f'{param}_psdiffs.pdf')

        plt.gcf().clear()

    def _plot_ps_table(param, index):
        zsamp_indices = np.linspace(0, zs.size, 4, dtype=int, endpoint=False)
        musamp_indices = np.linspace(0, mus.size, 2, dtype=int, endpoint=False)
        latexp = latex_paramdict[param]

        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        #with sns.plotting_context("paper"):
        fig, ax = plt.subplots(musamp_indices.size, zsamp_indices.size, sharex='col')
        fig.set_figheight(6)
        fig.set_figwidth(12)
        for mucount,muindex in enumerate(musamp_indices):
            for zcount,zindex in enumerate(zsamp_indices):
                fid = ps_bars_fid[:, zindex, muindex]
                updiff = np.abs(ups[index][:, zindex, muindex] - ps_bars_fid[:, zindex, muindex])
                downdiff = np.abs(downs[index][:, zindex, muindex] - ps_bars_fid[:, zindex, muindex])
                #_noise = noise[:, zindex, muindex]
                snr = ps_bars_fid[:, zindex, muindex]/noise[:, zindex, muindex]

                k_snr = ks[np.where( np.abs(snr - 1) < 0.1)]

                ax[mucount, zcount].plot(ks, ps_bars_fid[:, zindex, muindex], label=r"$\bar P({par})$".format(par=latexp))
                ax[mucount, zcount].plot(ks, updiff, label=r"$|\bar P({par} + \epsilon_{{ {par} }}) - \bar P({par})|$".format(par=latexp))
                ax[mucount, zcount].plot(ks, downdiff, label=r"$| \bar P({par} - \epsilon_{{ {par} }}) - \bar P({par}) |$".format(par=latexp), linestyle=':')
                ax[mucount, zcount].axvspan(k_snr[0], k_snr[-1], alpha=0.1, color='blue')
                ax[mucount, zcount].plot(ks, snr, label=r"$\bar P({par}) / \sqrt{{\sigma_P}}$".format(par=latexp))
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
        ax.flatten()[-2].legend(loc='upper center', bbox_to_anchor=(-0.2, -0.12), ncol=2)
        fig.tight_layout()
        fig.savefig(dir_ + figname + '_' + f'{param}_diff_table.svg')

#    for param in params.keys():
    muind = np.where(mus < 0.8)[0][0]
    _plot_ps_with_ratio('H0', param_index['H0'], 0, muind)

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

    DIR = 'userdata/figs/'

    params = np.load(args.params).item()
    psups = np.load(args.upfile)
    psdowns = np.load(args.downfile)
    prefacs = np.load(args.prefacfile)

    cc = get_cc(args.inifile)
    clst = Clustering(args.inifile, args.expname, args.gridname, args.version, cc)
    make_plots_upvdown(args.inifile, clst, psups, psdowns, prefacs, params, args.figname, DIR, args.legendloc)


if __name__ == '__main__':
    main()
