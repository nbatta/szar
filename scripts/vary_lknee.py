from __future__ import print_function
from builtins import str
import time
import os
import numpy as np

expList = ['S4-1.5-paper','S4-2.0-paper','S4-2.5-paper','S4-3.0-paper']


numCores = 50

gridName = "grid-default"

comp = "gen6"


lkneeList = np.arange(0,6000,500)
alphaList = [-4.,-4.5,-5.]

for exp in expList:

    for lknee in lkneeList:
        for alpha in alphaList:


            # do only sz
            cmd = "nohup wq sub -r \"mode:bycore;N:"+str(numCores)+";hostfile: auto;job_name: "+exp+";priority:med\" -c \"source ~/.bash_profile ; source ~/.bashrc ; cd ~/repos/szar ; mpirun -hostfile %hostfile% python bin/makeGrid.py "+exp+" "+gridName+" --skip-lensing -l "+str(lknee)+" -a "+str(alpha)+" \" > output"+str(time.time())+"_szgrid_"+exp+".log  &"


            print(cmd)
            os.system(cmd)
            time.sleep(0.3)

