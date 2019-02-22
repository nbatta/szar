#Python 2 compatibility stuff
from __future__ import print_function
from __future__ import division
from builtins import str
from builtins import range

import numpy as np
import argparse
from szar.derivatives import Derivs_Clustering, make_fisher, _make_fisher_cutks
from orphics.stats import FisherMatrix
from bin.futils import get_cc

from szar.clustering import Clustering

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outfile", help="name for output fisher file", required=True)
    parser.add_argument("-d", "--derivs", help="derivative file for fisher matrix", required=True)
    parser.add_argument("-f", "--factors", help="prefactor file for the fisher matrix", required=True)
    parser.add_argument("-p", "--params", help="parameters file for the fisher matrix", required=True)
    parser.add_argument("--maxkh", help="maximum value of kh to be used, explicitly cutting off the fisher. Note this is slower")
    parser.add_argument("--inifile", help="inifile for reading in a ClusterCosmology to get kh info for cutting off kh. Only needed if maxkh specified.")
    parser.add_argument("--expname", help="experiment name, only needed if maxkh specified")
    parser.add_argument("--gridname", help="name of SZ grid scheme, only needed if maxkh specified")
    args = parser.parse_args()

    DIR = 'userdata/'
    INIFILE = 'input/pipeline.ini'

    ps_derivs = np.load(args.derivs)
    ps_factors = np.load(args.factors)
    ps_params = np.load(args.params).item()
    ps_params = list(ps_params.keys())

    if args.maxkh is not None:
        cc = get_cc(args.inifile)
        clst = Clustering(args.inifile, args.expname, args.gridname, '0.6', cc)

        ps_fisher = _make_fisher_cutks(ps_derivs, ps_factors, clst, maxkh=0.14)
    else:
        ps_fisher = make_fisher(ps_derivs, ps_factors)

    ps_fisher = FisherMatrix(ps_fisher, ps_params)

    ps_fisher.to_pickle(DIR + 'fisher_' + args.outfile + '.pkl')

if __name__ == '__main__':
    main()
