import time
import os

expList = ['SO-3m','SO-5m','SO-6m','SO-7m','S4-3m','S4-5m','S4-6m','S4-7m','S4-5m-noatm','S4-6m-noatm','S4-7m-noatm','SO-5m-noatm','SO-6m-noatm','SO-7m-noatm','SO-3m-noatm','S4-3m-noatm']
#expList = ['S4-7m']

numCores = 20

#gridName = "grid-owl1"
grids = ["grid-owl2","grid-owl1"]
#grids = ["grid-fine-owl2"]


for exp in expList:
    for gridName in grids:
        
        # do only sz gen6 only
        #cmd = "nohup wq sub -r \"mode:bycore;N:"+str(numCores)+";hostfile: auto;job_name: szowl_"+exp+"_"+";group:[gen6];priority:med\" -c \"source ~/.bash_profile ; source ~/.bashrc ; cd ~/repos/SZ_filter ; mpirun -hostfile %hostfile% python bin/makeGrid.py "+exp+" "+gridName+" --skip-lensing \" > output"+str(time.time())+"_szgrid_"+exp+".log  &"

        # do only sz
        cmd = "nohup wq sub -r \"mode:bycore;N:"+str(numCores)+";hostfile: auto;job_name: szowl_"+exp+"_"+";priority:med\" -c \"source ~/.bash_profile ; source ~/.bashrc ; cd ~/repos/SZ_filter ; mpirun -hostfile %hostfile% python bin/makeGrid.py "+exp+" "+gridName+" --skip-lensing \" > output"+str(time.time())+"_szgrid_"+exp+".log  &"


        print cmd
        os.system(cmd)
        time.sleep(0.3)

