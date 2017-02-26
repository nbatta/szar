import time
import os

numParams = 1
numCores = 2*numParams+1
expList = ['SO-5m']
calList = ['CMB_all']
#calList = ['testCal']

for exp in expList:

    for cal in calList:


        massGridName = "data/"+exp+cal+".pkl"

        cmd = " mpirun -np " +str(numCores) + " python bin/makeWaDeriv.py "+exp+" "+cal+" "+massGridName+" data/forDerivsStep0.1 0.1 > output"+str(time.time())+".log  2>&1 &"

        # cmd = "nohup wq sub -r \"mode:bycore;N:"+str(numCores)+";hostfile: auto;job_name: ohhaithere;priority:med\" -c \"source ~/.bash_profile ; source ~/.bashrc ; cd ~/repos/SZ_filter ; mpirun -hostfile %hostfile% python bin/makeDerivs.py allParams "+exp+" "+cal+" "+massGridName+" \" > output"+str(time.time())+".log  &"
        #cmd = "nohup wq sub -r \"mode:bycore;N:"+str(numCores)+";hostfile: auto;job_name: ohhaithere;priority:med\" -c \"source ~/.bash_profile ; source ~/.bashrc ; cd ~/repos/SZ_filter ; mpirun -hostfile %hostfile% python bin/makeDerivs.py w0 "+exp+" "+cal+" "+massGridName+" \" > output"+str(time.time())+".log  &"
        
        print cmd
        os.system(cmd)
        time.sleep(0.3)

