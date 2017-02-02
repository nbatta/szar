import os
import sys
import numpy as np
import time

#zlist = np.arange(0.01,2.0,0.01)
#zlist = np.arange(0.1,2.0,0.1)
#zlist = np.arange(0.05,2.05,0.05)
#zlist = np.arange(2.05,3.05,0.05)
#zlist = np.arange(0.1,3.0,0.1)+0.1
zlist = np.arange(0.,3.0,0.05)+0.05
print "starting " , len(zlist),  " jobs."
for z in zlist:

    #cmd = "quick_wq.sh python scripts/gridMZ.py "+str(z)+" & "
    cmd = "quick_wq.sh python scripts/gridMZLens.py "+str(z)+" & "
    print cmd
    os.system(cmd)
    time.sleep(0.3)
