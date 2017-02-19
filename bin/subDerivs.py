import time
import os

numParams = 12
numCores = 2*numParams+1
expList = ['SO-5m','SO-6m','SO-7m','S4-5m','S4-6m','S4-7m']
calList = ['CMBAll','CMBPol']#,'OWL1','OWL2','CMBAllOWL1','CMBAllOWL2','CMBPolOWL1','CMBPolOWL2']
#calList = ['testCal']

for exp in expList:

    for cal in calList:


        massGridName = "data/"+exp+cal+".pkl"


        cmd = "nohup wq sub -r \"mode:bycore;N:"+str(numCores)+";hostfile: auto;job_name: takesAboutInfinityMinutes;priority:med\" -c \"source ~/.bash_profile ; source ~/.bashrc ; cd ~/repos/SZ_filter ; mpirun -hostfile %hostfile% python bin/makeDerivs.py allParams "+exp+" "+cal+" "+massGridName+" \" > output"+str(time.time())+".log  &"

        print cmd
        os.system(cmd)
        time.sleep(0.3)

