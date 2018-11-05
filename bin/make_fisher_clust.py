#Python 2 compatibility stuff
from __future__ import print_function
from __future__ import division
from builtins import str
from builtins import range
from six.moves import configparser
import six
if six.PY2:
  ConfigParser = configparser.SafeConfigParser
else:
  ConfigParser = configparser.ConfigParser

import numpy as np
import argparse

from orphics.io import FisherPlots
from szar.derivatives import Derivs_Clustering, make_fisher

def fisher_validation_tests(ups, downs, steps, derivs, prefacs):
    print("Running some checks on the fisher matrix...")
    print("\n")

    assert np.array_equal((ups - downs)/(2 * steps[..., np.newaxis, np.newaxis, np.newaxis]), derivs)
    derivmax = np.abs(derivs).max()
    derivmin = np.abs(derivs).min()
    print(f"the max derivative abval is {np.format_float_scientific(derivmax)}")
    print(f"the min derivative abval is {derivmin}")

    nparams,nks,nzs,nmus = derivs.shape
    prefacmax = np.abs(prefacs).max()
    print(f"the max prefactor abval is {np.format_float_scientific(prefacmax)}")
    print(f"fisher is summed over {nks} k-bins, {nzs} z-bins, and {nmus} mu-bins")
    print(f"thus an upper bound for the max value of the fisher components is {np.format_float_scientific(nks * nmus * nzs * derivmax**2 * prefacmax)}")

    fishermat = make_fisher(derivs, prefacs)
    print(f"the actual max value of the fisher is {np.abs(fishermat).max()}")

    print("\n")

    print("creating a test fisher to see if sum is being done right...")
    a = np.ones((nparams, nks, nzs, nmus))
    b = np.ones((nks, nzs, nmus))
    testfisher = make_fisher(a, b)
    theoryval = nks * nzs * nmus
    print(f"the max value of the test fisher SHOULD be {theoryval}")
    print(f"the actual max value of the test fisher is {testfisher.max()}")

def make_constraint_curves_orphics(config, fishmat, expname, gridname, version):
    fishSection = "lcdm"
    paramList = config.get('fisher-'+fishSection,'paramList').split(',')
    paramLatexList = config.get('fisher-'+fishSection,'paramLatexList').split(',')

    deriv = Derivs_Clustering(INIFILE, expname, gridname, version)
    deriv.instantiate_params()
    fparams = deriv.params

    fplots = FisherPlots()
    fplots.startFig()
    fplots.addSection(fishSection,paramList,paramLatexList,fparams)
    fplots.addFisher(fishSection,'AAAAAAB', fishmat)

    fplots.plotTri(fishSection,paramList,['AAAAAAB'],labels=['S4'],saveFile="figs/ellipses_test.png",loc='best')

if __name__ == '__main__':
    #Reads in experimental and grid parameters for obtaining derivatives files
    parser = argparse.ArgumentParser()
    parser.add_argument("expname", help="name of experiment")
    parser.add_argument("gridname", help="name of grid")
    parser.add_argument("version", help="version number")
    args = parser.parse_args()
    saveid = args.expname + "_" + args.gridname + "_v" + args.version + '_'

    DIR = "datatest/"
    TIMESTAMP = "_2018-11-05-10-26-09-EST"
    DERIVSFILE = saveid + "fish_derivs" + TIMESTAMP + '.npy'
    FACTORFILE = saveid + "fish_factor" + TIMESTAMP + '.npy'
    UPFILE = saveid + "psups" + TIMESTAMP + '.npy'
    DOWNFILE = saveid + "psdowns" + TIMESTAMP + '.npy'
    STEPFILE = saveid + "steps" + TIMESTAMP + '.npy'

    fisher_derivs = np.load(DIR + DERIVSFILE)
    fisher_facs = np.load(DIR + FACTORFILE)
    ps_ups = np.load(DIR + UPFILE)
    ps_downs = np.load(DIR + DOWNFILE)
    steps = np.load(DIR + STEPFILE)

    config = ConfigParser()
    config.optionxform=str
    INIFILE = "input/pipeline.ini"
    config.read(INIFILE)

    fisher = make_fisher(fisher_derivs, fisher_facs)
    fisher = fisher[:-1]
    fisher = fisher.T[:-1].T
    
    make_constraint_curves(config, fisher, args.expname, args.gridname, args.version) 
