import time
import os


expList = ['SO-5m','SO-6m','SO-7m','S4-5m','S4-6m','S4-7m']
#expList = ['SO-6m','SO-7m','S4-5m','S4-6m','S4-7m']
calList = ['CMB_all_nodelens','CMB_pol_nodelens','CMB_all','CMB_pol']#,'OWL1','OWL2','CMBAllOWL1','CMBAllOWL2','CMBPolOWL1','CMBPolOWL2']
#expList = ['S4-7m']
#calList = ['CMB_all']#,'OWL1','OWL2','CMBAllOWL1','CMBAllOWL2','CMBPolOWL1','CMBPolOWL2']

numCores = 130

for exp in expList:
    for cal in calList:




        cmd = "nohup wq sub -r \"mode:bycore;N:"+str(numCores)+";hostfile: auto;job_name: ohhaithere;priority:med\" -c \"source ~/.bash_profile ; source ~/.bashrc ; cd ~/repos/SZ_filter ; mpirun -hostfile %hostfile% python bin/makeCMBGrid.py "+exp+" "+cal+" \" > output"+str(time.time())+".log  &"

        print cmd
        os.system(cmd)
        time.sleep(0.3)

