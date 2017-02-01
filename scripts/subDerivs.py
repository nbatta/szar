import os
import sys
import numpy as np
import time


paramList =  ['H0','ombh2','omch2','As','ns','mnu','b_ym','alpha_ym','Ysig','gamma_ym','beta_ym']  

for p in paramList:

    cmd = "python tests/makeDerivs.py "+str(p)+" & "
    print cmd
    os.system(cmd)
    time.sleep(0.3)
