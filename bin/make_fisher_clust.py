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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--derivs", help="derivative files for fisher matrix")
    parser.add_argument("-f", "--factors", help="prefactors for the fisher matrix")
    args = parser.parse_args()

    fisher = get_fisher(fisherfile)

if __name__ == '__main__':
    main()
