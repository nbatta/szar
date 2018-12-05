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
from make_fisher_clust import load_fisher, _get_params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fisher", help="input fisher matrix")
    parser.add_argument("--array", help="indicate fisher is not a FisherMatrix object but a tuple of params,ndarray", action="store_true")
    args = parser.parse_args()

    fisher_file = args.fisher
    fisher = load_fisher(fisher_file)

    if not args.array:
        fisher_params = fisher.columns.values
    else:
        fisher_params,fisher = fisher
        fisher = FisherMatrix(fisher, fisher_params)

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

    print(constraints)

if __name__ == '__main__':
    main()
