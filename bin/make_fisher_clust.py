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

if __name__ == '__main__':
    DIR = "datatest/"
    DERIVSFILE = "S4-1.0-CDT_grid-owl2_v0.6fish_derivs_1-11-2018.npy"
    FACTORFILE = "S4-1.0-CDT_grid-owl2_v0.6fish_factor_1-11-2018.npy"

    fisher_derivs = np.load(DIR + DERIVSFILE)
    fisher_facs = np.load(DIR + FACTORFILE)

    print(make_fisher(fisher_derivs, fisher_facs))
