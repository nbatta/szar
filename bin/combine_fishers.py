#Python 2 compatibility stuff
from __future__ import print_function
from __future__ import division
from builtins import str
from builtins import range

import string
import numpy as np
import argparse
from futils import load_fisher
from orphics.stats import FisherMatrix

def _load_and_process(fisher_files):
    fisher_list = []
    for file_ in fisher_files:
        fisher = load_fisher(file_)
        fisher_list.append(fisher)

    return fisher_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fisher_files", help="fisher matrix file locations", nargs='+')
    parser.add_argument('-o', '--outfile', help="name of file for the summed fishers", default="datatest/fisher_sum.pkl")
    args = parser.parse_args()

    if len(args.fisher_files) == 1:
        print("You need to specify more than one fisher matrix.")
        sys.exit()

    fisher_list = _load_and_process(args.fisher_files)

    fisher_sum = np.sum(fisher_list)

    assert type(fisher_sum) is FisherMatrix

    fisher_sum.to_pickle(args.outfile)

def test_add():
    fisher1 = np.ones((10,10))
    fisher2 = np.ones((10,10))
    fisher3 = 200 * np.ones((5,5))

    params1 = list(string.ascii_lowercase)[:10]
    params2 = params1
    params3 = list(string.ascii_lowercase)[1:6]
    params3 = params3[::-1]
    
    fisher1 = FisherMatrix(fisher1, params1)
    fisher2 = FisherMatrix(fisher2, params2)
    fisher3 = FisherMatrix(fisher3, params3)
    
    fisher_sum = np.sum([fisher1, fisher2, fisher3])
    print(f"{fisher1}, \n{fisher2}, and \n{fisher3} \n\n summed to \n\n {fisher_sum}")

def test_load_and_process():
    testfisher = np.ones((10,10))
    params = list(string.ascii_lowercase)[:10]

    # Test pickled FM objects
    testfisher_FMObj_out = FisherMatrix(testfisher, params)
    outfile_FMObj = 'datatest/testfisher_FMObj.pkl'
    testfisher_FMObj_out.to_pickle(outfile_FMObj)
    testfisher_FMObj_in = load_fisher(outfile_FMObj)
    assert testfisher_FMObj_out.equals(testfisher_FMObj_in)

    #test .txt files with hashed params
    outfile_txt_hashparams = 'datatest/testfisher_txt_hashparams.txt'
    np.savetxt(outfile_txt_hashparams, testfisher, delimiter=',', header=",".join(params))
    testfisher_txt_hashparams_in = load_fisher(outfile_txt_hashparams)
    assert testfisher_FMObj_out.equals(testfisher_txt_hashparams_in)

    #test .txt files that are space separated
    outfile_txt_spacesep = 'datatest/testfisher_txt_spacesep.txt'
    np.savetxt(outfile_txt_spacesep, testfisher, delimiter=' ', header=",".join(params))
    testfisher_txt_spacesep_in = load_fisher(outfile_txt_spacesep)
    assert testfisher_FMObj_out.equals(testfisher_txt_spacesep_in)

    print("Loading checks passed")

if __name__ == '__main__':
    #test_load_and_process()
    #test_add()
    main()
