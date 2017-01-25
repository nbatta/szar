import os
import sys
import numpy as np
import time

zlist = np.arange(0.01,2.0,0.01)
print "starting " , len(zlist),  " jobs."
for z in zlist:

    cmd = "quick_wq.sh python scripts/gridMZ.py "+str(z)+" & "
    print cmd
    os.system(cmd)
    time.sleep(0.3)
