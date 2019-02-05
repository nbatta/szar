#Python 2 compatibility stuff
from __future__ import print_function
from __future__ import division
from builtins import str
from builtins import range

import numpy as np
import argparse
from szar.derivatives import Derivs_Clustering, make_fisher
from orphics.stats import FisherMatrix

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outfile", help="name for output fisher file")
    parser.add_argument("-d", "--derivs", help="derivative file for fisher matrix")
    parser.add_argument("-f", "--factors", help="prefactor file for the fisher matrix")
    parser.add_argument("-p", "--params", help="parameters file for the fisher matrix")
    args = parser.parse_args()

    DIR = 'datatest/'
    INIFILE = 'input/pipeline.ini'

    ps_derivs = np.load(args.derivs)
    ps_factors = np.load(args.factors)
    ps_params = np.load(args.params).item()
    ps_params = list(ps_params.keys())

    ps_fisher = make_fisher(ps_derivs, ps_factors)
    ps_fisher = FisherMatrix(ps_fisher, ps_params)

    #ps_fisher.delete('b_wl') #should replace with function to detect zeroed columns + rows 

    ps_fisher.to_pickle(DIR + 'fisher_' + args.outfile + '.pkl')

if __name__ == '__main__':
    main()
