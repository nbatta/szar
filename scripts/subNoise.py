import os
import sys
import numpy as np
import time

beam = sys.argv[1]
noise = sys.argv[2]

for Mexp in np.arange(14.0,15.4,0.2):
    for z in np.arange(0.1,0.8,0.2):

        cmd = "quick_wq.sh python scripts/gridMZ.py "+str(beam)+" "+str(noise)+" "+str(Mexp)+" "+str(z)+" & "
        print cmd
        os.system(cmd)
        time.sleep(0.2)
