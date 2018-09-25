import numpy as np
from szar.counts import ClusterCosmology
from configparser import ConfigParser
from orphics.io import dict_from_section,list_from_config
from szar.clustering import Clustering
import sys

INIFILE = "input/pipeline.ini"
expName = 'S4-1.0-CDT'
gridName = 'grid-owl2'
version = '0.6'

clst = Clustering(INIFILE,expName,gridName,version)

mus = np.array([0])
ks = clst.HMF.kh
zs = clst.HMF.zarr
try:
    pstildes = clst.ps_tilde(mus)
except Exception as e:
    print("Test failed at clustering.ps_tilde")
    print(e)
    sys.exit()


expected = np.outer(zs, ks)

if pstildes.shape != expected.shape:
    print("pstildes shape is not the expected value; test failed!")
else:
    print("Tests passed!")

