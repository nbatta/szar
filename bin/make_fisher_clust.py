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

def make_fisher(derivs, prefactors):
    fisher_terms = prefactors[np.newaxis, ...] * derivs
    fisher_mat = np.einsum('aijk,bijk->ab', derivs, fisher_terms)
    return fisher_mat

def fisher_validation_tests(ups, downs, steps, derivs, prefacs):
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

if __name__ == '__main__':
    DIR = "datatest/"
    DERIVSFILE = "S4-1.0-CDT_grid-owl2_v0.6_fish_derivs_2018-11-02-14-56-36-EDT.npy"
    FACTORFILE = "S4-1.0-CDT_grid-owl2_v0.6_fish_factor_2018-11-02-14-56-36-EDT.npy"
    UPFILE = "S4-1.0-CDT_grid-owl2_v0.6_psups_2018-11-02-14-56-36-EDT.npy"
    DOWNFILE = "S4-1.0-CDT_grid-owl2_v0.6_psdowns_2018-11-02-14-56-36-EDT.npy"
    STEPFILE = "S4-1.0-CDT_grid-owl2_v0.6_steps_2018-11-02-14-56-36-EDT.npy"


    fisher_derivs = np.load(DIR + DERIVSFILE)
    fisher_facs = np.load(DIR + FACTORFILE)
    ps_ups = np.load(DIR + UPFILE)
    ps_downs = np.load(DIR + DOWNFILE)
    steps = np.load(DIR + STEPFILE)

    fisher_validation_tests(ps_ups, ps_downs, steps, fisher_derivs, fisher_facs)
