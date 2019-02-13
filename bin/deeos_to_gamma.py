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
import os
import sys
import pickle
import time

from szar.derivatives import Derivs_Clustering, make_fisher
from orphics.stats import FisherMatrix
from pandas import DataFrame
from futils import load_fisher

def get_gamma_constraint(fisher, fisher_params):
    #equation for gamma is gamma = gamma0 + b(1 + w0 + wa/2) with b=0.05

    b = 0.05
    dw0_dgamma = 1/b
    dwa_dgamma = 2/b

    jw0 = dw0_dgamma
    jwa = dwa_dgamma

    jacob_arr = np.zeros(fisher.shape)
    jacob_arr = np.delete(jacob_arr, -1, axis=1)

    rows = fisher_params
    columns = []
    for param in fisher_params:
        if param == 'w0':
            continue
        if param == 'wa':
            columns.append('gamma')
            continue
        else:
            columns.append(param)

    jacobian = DataFrame(jacob_arr, columns=columns, index=rows)

    for param in columns:
        if param != 'gamma':
            jacobian[param][param] = 1

    jacobian['gamma']['w0'] = jw0
    jacobian['gamma']['wa'] = jwa

    newfisher_arr = jacobian.values.T @ fisher.values @ jacobian.values

    newfisher = FisherMatrix(newfisher_arr, columns)

    constraints = newfisher.sigmas()

    return constraints['gamma']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fisher", help="input fisher matrix")
    args = parser.parse_args()

    fisher_file = args.fisher
    fisher = load_fisher(fisher_file)

    fisher_params = fisher.columns.values

    sigma_gamma = get_gamma_constraint(fisher, fisher_params)
    print(sigma_gamma)

if __name__ == '__main__':
    main()
