import time
import os

numParams = 12
#numParams = 1
numCores = 2*numParams+1
expList = ['SO-5m-noatm']#,'SO-6m-noatm','SO-7m-noatm']
calList = ['CMB_all']#,'CMB_all_nodelens','CMB_pol_nodelens']#,'OWL1','OWL2','CMBAllOWL1','CMBAllOWL2','CMBPoexpList = ['SO-5m','SO-6m','SO-7m','S4-5m','S4-6m','S4-7m']
#calList = ['CMB_all','CMB_pol','CMB_all_nodelens','CMB_pol_nodelens']#,'OWL1','OWL2','CMBAllOWL1','CMBAllOWL2','CMBPolOWL1','CMBPolOWL2']
# expList = ['SO-5m']
# calList = ['testGrid']

for exp in expList:

    for cal in calList:


        massGridName = "data/"+exp+cal+".pkl"

        cmd = " mpirun -np " +str(numCores) + " python bin/makeDerivs.py allParams "+exp+" "+cal+" "+massGridName+"  > output"+str(time.time())+".log  2>&1 &"

        # cmd = "nohup wq sub -r \"mode:bycore;N:"+str(numCores)+";hostfile: auto;job_name: ohhaithere;priority:med\" -c \"source ~/.bash_profile ; source ~/.bashrc ; cd ~/repos/SZ_filter ; mpirun -hostfile %hostfile% python bin/makeDerivs.py allParams "+exp+" "+cal+" "+massGridName+" \" > output"+str(time.time())+".log  &"
        #cmd = "nohup wq sub -r \"mode:bycore;N:"+str(numCores)+";hostfile: auto;job_name: ohhaithere;priority:med\" -c \"source ~/.bash_profile ; source ~/.bashrc ; cd ~/repos/SZ_filter ; mpirun -hostfile %hostfile% python bin/makeDerivs.py w0 "+exp+" "+cal+" "+massGridName+" \" > output"+str(time.time())+".log  &"
        
        print cmd
        os.system(cmd)
        time.sleep(0.3)

