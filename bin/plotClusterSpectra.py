import numpy as np
from szar.counts import ClusterCosmology
from configparser import ConfigParser
from orphics.io import dict_from_section,list_from_config
from szar.clustering import Clustering
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
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

def make_plots_upvdown(clst, ups, downs, factors, figname, legendloc):
    mus = np.linspace(-1,1,9)
    ks = clst.HMF.kh
    delta_ks = np.diff(ks)
    
    ps_bars_fid = clst.fine_ps_bar(mus)

    plt.plot(ks, ps_bars_fid[:,0,0], label=r"fiducial")
    plt.plot(ks, ups[0][:,0,0], label=r"up")
    plt.plot(ks, downs[0][:,0,0], label=r"down")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc=legendloc)

    plt.savefig(figname)
    plt.gcf().clear()

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
    args = parser.parse_args()

    inifile = args.inifile
    expname = args.expname
    gridname = args.gridname
    version = args.version
    figname = args.figname
    legendloc = args.legendloc 

    psups = np.load(args.upfile)
    psdowns = np.load(args.downfile)
    prefacs = np.load(args.prefacfile)
    
    cc = get_cc(inifile)
    clst = Clustering(inifile,expname,gridname,version,cc)
    make_plots_upvdown(clst, psups, psdowns, prefacs, figname, legendloc)


if __name__ == '__main__':
    main()
