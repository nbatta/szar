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

#cosmo imports
from orphics.io import dict_from_section, list_from_config
import numpy as np
from szar.derivatives import Derivs_Clustering

# etc
import argparse
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("expname", help="name of experiment")
    parser.add_argument("gridname", help="name of grid")
    parser.add_argument("version", help="version number")
    args = parser.parse_args()

    INIFILE = "input/pipeline.ini"
    DIR = "datatest/"
    FISH_FAC_NAME = "fish_factor"
    FISH_DERIV_NAME = "fish_derivs" 
    UPNAME = "psups"
    DOWNNAME = "psdowns"
    STEPNAME = "steps"
    PARAMSNAME = "params"
    currenttime = time.strftime("%Y-%m-%d-%H-%M-%S-%Z", time.localtime())

    deriv = Derivs_Clustering(INIFILE, args.expname, args.gridname, args.version)
    deriv.instantiate_params()
    deriv.instantiate_grids()

    fish_derivs, fish_facs, ups, downs = deriv.make_derivs()

    np.save(DIR + deriv.saveid + '_' + FISH_FAC_NAME + '_' + currenttime, fish_facs)
    np.save(DIR + deriv.saveid + '_' + FISH_DERIV_NAME + '_' + currenttime, fish_derivs)
    np.save(DIR + deriv.saveid + '_' + UPNAME + '_' + currenttime, ups)
    np.save(DIR + deriv.saveid + '_' + DOWNNAME + '_' + currenttime, downs)
    np.save(DIR + deriv.saveid + '_' + STEPNAME + '_' + currenttime, deriv.steps)
    np.save(DIR + deriv.saveid + '_' + PARAMNAME + '_' + currenttime, deriv.fisher_params)
