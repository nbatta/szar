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
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import seaborn as sns
sns.set(style='whitegrid', font_scale=1.5, rc={"lines.linewidth": 2,'lines.markersize': 8.0,})

from orphics.io import FisherPlots
from szar.derivatives import Derivs_Clustering, make_fisher
from orphics.stats import FisherMatrix

def load_fisher(file_):
    filename, file_extension = os.path.splitext(file_)

    if file_extension == '.csv':
        fisher = np.loadtxt(file_, delimiter=' ')
    elif file_extension == '.npy':
        fisher = np.load(file_)
    elif file_extension == '.pkl':
        with open(file_, 'rb') as pickle_file:
            try:
                fisher = pickle.load(pickle_file)
            except UnicodeDecodeError:
                fisher = pickle.load(pickle_file, encoding='latin1')
    else:
        print(f"Filetype of extra fisher file {file_} not supported")
        sys.exit()
    
    return fisher

def _get_params(fisherfile):
    filename, file_extension = os.path.splitext(fisherfile)
    known_params = {'data/Feb18_FisherMat_Planck_tau0.01_lens_fsky0.6.csv':['H0', 'ombh2', 'omch2', 'tau', 'As', 'ns', 'mnu']}

    if fisherfile in known_params:
        return known_params[fisherfile]
    else:
        print(f"Sorry, the parameters for {fisherfile} are not known")
        sys.exit()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outfile", help="name for output fisher file")
    parser.add_argument("-d", "--derivs", help="derivative file for fisher matrix")
    parser.add_argument("-f", "--factors", help="prefactor file for the fisher matrix")
    parser.add_argument("-p", "--params", help="parameters file for the fisher matrix")
    parser.add_argument("--extra-fishers", help="extra fisher matrix files to be added to clustering", nargs='*')
    args = parser.parse_args()

    DIR = 'datatest/'
    INIFILE = 'input/pipeline.ini'

    ps_derivs = np.load(args.derivs)
    ps_factors = np.load(args.factors)
    ps_params = np.load(args.params).item()
    ps_params = list(ps_params.keys())

    ps_fisher = make_fisher(ps_derivs, ps_factors)
    ps_fisher = FisherMatrix(ps_fisher, ps_params)

    if args.extra_fishers:
        extra_fishers = []
        for fisherfile in args.extra_fishers:
            fishmat = load_fisher(fisherfile)

            if type(fishmat) is tuple:
                params,fishmat = fishmat
                assert type(fishmat) is np.ndarray
            else:
                params = _get_params(fisherfile)

            fishmat = FisherMatrix(fishmat, params)
                
            extra_fishers.append(fishmat)

    full_fisher = ps_fisher.copy()

    old_constraints = extra_fishers[0].sigmas().copy()

    if extra_fishers:
        for extra in extra_fishers:
            full_fisher = full_fisher + extra

    #should replace with function to detect zeroed columns + rows
    full_fisher.delete('b_wl')

    currenttime = time.strftime("%Y-%m-%d-%H-%M-%S-%Z", time.localtime())

    if args.outfile is not None:
        full_fisher.to_pickle(DIR + 'fisher_' + args.outfile + '_' + currenttime + '.pkl')

    for key,val in full_fisher.sigmas().items():
        print(f"{key}: {val}\n % improvement over abundances + Planck: {100 * (1 - val/old_constraints[key])}")
    
    cov = np.linalg.inv(full_fisher.values)
    cov = FisherMatrix(cov, full_fisher.columns.values)
    
    if args.outfile is not None:
        cov.to_pickle(DIR + 'covariance_' + args.outfile + currenttime + '_' + '.pkl')

if __name__ == '__main__':
    main()
