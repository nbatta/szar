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

def _add_fishers(fisher_list):
    zeros = np.array([[0,0],[0,0]], dtype=float)
    dummyparam1 = 'HiMyNameIsPaulTheDummyParamISureHopeThisIsntARealParamOfYourFisher'
    dummyparam2 = 'HiMyNameIsPaulaTheDummyParamISureHopeThisIsntARealParamOfYourFisherEither'
    dummies = [dummyparam1, dummyparam2]
    total = FisherMatrix(zeros, dummies)
    
    for fish in fisher_list:
        total = total + fish

    total.delete(dummyparam1)
    total.delete(dummyparam2)
    return total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fisher_files", help="fisher matrix file locations", nargs='+')
    parser.add_argument('-o', '--outfile', help="name of file for the summed fishers", default="userdata/fisher_sum.pkl")
    args = parser.parse_args()

    if len(args.fisher_files) == 1:
        print("You need to specify more than one fisher matrix.")
        sys.exit()

    fisher_list = _load_and_process(args.fisher_files)
    
    fisher_sum = _add_fishers(fisher_list)

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
    
    fisher_sum = _add_fishers([fisher1, fisher2, fisher3])
    print(f"{fisher1}, \n{fisher2}, and \n{fisher3} \n\n summed to \n\n {fisher_sum}")

def test_load_and_process():
    testfisher = np.ones((10,10))
    params = list(string.ascii_lowercase)[:10]

    # Test pickled FM objects
    testfisher_FMObj_out = FisherMatrix(testfisher, params)
    outfile_FMObj = 'userdata/testdata/testfisher_FMObj.pkl'
    testfisher_FMObj_out.to_pickle(outfile_FMObj)
    testfisher_FMObj_in = load_fisher(outfile_FMObj)
    assert testfisher_FMObj_out.equals(testfisher_FMObj_in)

    #test .txt files with hashed params
    outfile_txt_hashparams = 'userdata/testdata/testfisher_txt_hashparams.txt'
    np.savetxt(outfile_txt_hashparams, testfisher, delimiter=',', header=",".join(params))
    testfisher_txt_hashparams_in = load_fisher(outfile_txt_hashparams)
    assert testfisher_FMObj_out.equals(testfisher_txt_hashparams_in)

    #test .txt files that are space separated
    outfile_txt_spacesep = 'userdata/testdata/testfisher_txt_spacesep.txt'
    np.savetxt(outfile_txt_spacesep, testfisher, delimiter=' ', header=",".join(params))
    testfisher_txt_spacesep_in = load_fisher(outfile_txt_spacesep)
    assert testfisher_FMObj_out.equals(testfisher_txt_spacesep_in)

    print("Loading checks passed")

def test_weird_file_add():
    test_files = ['userdata/testdata/testfisher1.txt',
                  'userdata/testdata/testfisher2.txt',
                  'userdata/testdata/testfisher3.txt']

    test_fishers = _load_and_process(test_files)

    sum_fisher = _add_fishers(test_fishers)

    check_vals = np.ones((6,6))
    check_vals += np.array([[0,0,0,0,0,0], [0,2,2,0,0,2], [0,2,2,0,0,2], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,2,2,0,0,2]])
    check_vals += np.array([[0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,10]])

    check_params = ['A','B','C','D','E','F']
    check_fisher = FisherMatrix(check_vals, check_params)

    try:
        assert check_fisher.equals(sum_fisher)
        print("Test of weird_file_add passed")
    except AssertionError:
        print("Test of weird_file_add failed!")
        print("Expected to get:")
        print(check_fisher)
        print("got instead")
        print(sum_fisher)

    another_small_fisher = FisherMatrix(np.array([[100,0],[0,0]]), ['F','A'])
    asf_file = 'userdata/testdata/testfisher4.pkl'
    another_small_fisher.to_pickle(asf_file)

    asf_loaded = _load_and_process([asf_file])
    newlist = [sum_fisher] + asf_loaded
    newsum = _add_fishers(newlist)

    check_fisher += np.array([[0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,100]])

    try:
        assert check_fisher.equals(newsum)
        print("Test of weird_file_add passed")
    except AssertionError:
        print("Test of weird_file_add failed!")
        print("Expected to get:")
        print(check_fisher)
        print("got instead")
        print(newsum)

if __name__ == '__main__':
    #test_load_and_process()
    #test_add()
    #test_weird_file_add()
    main()
