import os
import sys
import pickle
import numpy as np
from orphics.stats import FisherMatrix

def _get_header(file_):
    with open(file_) as filein:
        header = filein.readline()
        header = header.lstrip('#')
        header = header.rstrip('\n')
        header = header.replace(' ', '')
        header = header.split(',')
    return header

def load_fisher(file_):
    filename, file_extension = os.path.splitext(file_)

    if file_extension == '.txt':
        try:
            fisher = np.loadtxt(file_, delimiter=',')
        except ValueError:
            fisher = np.loadtxt(file_, delimiter=' ')

        params = _get_header(file_)
        fisher = FisherMatrix(fisher, params)

    elif file_extension == '.pkl':
        try:
            with open(file_, 'rb') as pickle_file:
                fisher = pickle.load(pickle_file)
        except UnicodeDecodeError:
            with open(file_, 'rb') as pickle_file:
                fisher = pickle.load(pickle_file, encoding='latin1')

        fisher.params = fisher.columns.values.tolist()
    else:
        print(f"Filetype of extra fisher file {file_} not supported")
        sys.exit()
    assert fisher.params is not None
    assert type(fisher) is FisherMatrix
    return fisher
