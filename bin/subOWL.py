import time
import os

numParams = 12
#numParams = 1
numCores = 2*numParams+1
expList = ['S4-3m-noatm','S4-5m-noatm','S4-6m-noatm','S4-7m-noatm','SO-3m-noatm','SO-5m-noatm','SO-6m-noatm','SO-7m-noatm']#,'SO-6m','SO-7m','S4-5m','S4-6m','S4-7m']
#'SO-5m-noatm','SO-6m-noatm','SO-7m-noatm',
#expList = ['AdvAct']
calList = ['owl2','owl1']#,'owl1']

for exp in expList:

    for cal in calList:


        massGridName = "data/"+cal+".pkl"

        #cmd = "nohup wq sub -r \"mode:bycore;N:"+str(numCores)+";hostfile: auto;job_name: ohhaithere;priority:med\" -c \"source ~/.bash_profile ; source ~/.bashrc ; cd ~/repos/SZ_filter ; mpirun -hostfile %hostfile% python bin/makeWaDeriv.py "+exp+" "+cal+" "+massGridName+" data/forDerivsStep0.1 0.1 \" > output"+str(time.time())+".log  &"
        #cmd = "mpirun -np "+str(numCores)+" python bin/makeDerivs.py w0 "+exp+" "+cal+" "+massGridName+"  > output"+str(time.time())+".log  &"
        #cmd = "mpirun -np "+str(numCores)+" python bin/makeDerivs.py allParams "+exp+" "+cal+" "+massGridName+"  > output"+str(time.time())+".log  &"

        cmd = "nohup wq sub -r \"mode:bycore;N:"+str(numCores)+";hostfile: auto;job_name: "+exp+"_"+cal+";priority:med\" -c \"source ~/.bash_profile ; source ~/.bashrc ; cd ~/repos/SZ_filter ; mpirun -hostfile %hostfile% python bin/makeDerivs.py allParams "+exp+" "+cal+" "+massGridName+" \" > output"+str(time.time())+"_owlderiv_"+exp+"_"+cal+".log  &"
        #cmd = "nohup wq sub -r \"mode:bycore;N:"+str(numCores)+";hostfile: auto;job_name: ohhaithere;priority:med\" -c \"source ~/.bash_profile ; source ~/.bashrc ; cd ~/repos/SZ_filter ; mpirun -hostfile %hostfile% python bin/makeDerivs.py w0 "+exp+" "+cal+" "+massGridName+" \" > output"+str(time.time())+".log  &"

        print cmd
        os.system(cmd)
        time.sleep(0.3)

