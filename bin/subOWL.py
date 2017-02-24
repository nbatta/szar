import time
import os

#numParams = 12
numParams = 1
numCores = 2*numParams+1
expList = ['SO-5m-noatm','SO-6m-noatm','SO-7m-noatm']#,'S4-5m','S4-6m','S4-7m']
#expList = ['AdvAct']
calList = ['owl2']#'owl1']

for exp in expList:

    for cal in calList:


        massGridName = "data/"+cal+".pkl"

        cmd = "mpirun -np "+str(numCores)+" python bin/makeDerivs.py w0 "+exp+" "+cal+" "+massGridName+"  > output"+str(time.time())+".log  &"
        #cmd = "mpirun -np "+str(numCores)+" python bin/makeDerivs.py allParams "+exp+" "+cal+" "+massGridName+"  > output"+str(time.time())+".log  &"

        #cmd = "nohup wq sub -r \"mode:bycore;N:"+str(numCores)+";hostfile: auto;job_name: ohhaithere;priority:med\" -c \"source ~/.bash_profile ; source ~/.bashrc ; cd ~/repos/SZ_filter ; mpirun -hostfile %hostfile% python bin/makeDerivs.py allParams "+exp+" "+cal+" "+massGridName+" \" > output"+str(time.time())+".log  &"
        #cmd = "nohup wq sub -r \"mode:bycore;N:"+str(numCores)+";hostfile: auto;job_name: ohhaithere;priority:med\" -c \"source ~/.bash_profile ; source ~/.bashrc ; cd ~/repos/SZ_filter ; mpirun -hostfile %hostfile% python bin/makeDerivs.py w0 "+exp+" "+cal+" "+massGridName+" \" > output"+str(time.time())+".log  &"

        print cmd
        os.system(cmd)
        time.sleep(0.3)

