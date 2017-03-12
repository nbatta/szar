import time
import os


#expList = ['AdvAct','SO-3m','SO-5m','SO-6m','SO-7m','S4-3m','S4-5m','S4-6m','S4-7m','S4-5m-noatm','S4-6m-noatm','S4-7m-noatm','SO-5m-noatm','SO-6m-noatm','SO-7m-noatm']
expList = ['S4-5m','S4-7m']
#expList = ['S4-5m-noatm','S4-6m-noatm','S4-7m-noatm']
#expList = ['AdvAct']
calList = ['CMB_all']#,'CMB_pol','CMB_all_nodelens','CMB_pol_nodelens']#,'CMB_pol']#,'OWL1','OWL2','CMBAllOWL1','CMBAllOWL2','CMBPolOWL1','CMBPolOWL#expList = ['SO-6m','SO-7m','S4-5m','S4-6m','S4-7m']
#calList = ['CMB_all_nodelens','CMB_pol_nodelens','CMB_all','CMB_pol']#,'OWL1','OWL2','CMBAllOWL1','CMBAllOWL2','CMBPolOWL1','CMBPolOWL2']
#expList = ['S4-7m']
#calList = ['CMB_all']#,'OWL1','OWL2','CMBAllOWL1','CMBAllOWL2','CMBPolOWL1','CMBPolOWL2']

numCores = 142

gridName = "grid-default"



for exp in expList:
    for cal in calList:

        # do both
        cmd = "nohup wq sub -r \"mode:bycore;N:"+str(numCores)+";hostfile: auto;job_name: "+exp+"_"+cal+";priority:med\" -c \"source ~/.bash_profile ; source ~/.bashrc ; cd ~/repos/SZ_filter ; mpirun -hostfile %hostfile% python bin/makeGrid.py "+exp+" "+gridName+" "+cal+"  \" > output"+str(time.time())+"_szgrid_"+exp+"_"+cal+".log  &"

        
        # do only sz
        #cmd = "nohup wq sub -r \"mode:bycore;N:"+str(numCores)+";hostfile: auto;job_name: "+exp+"_"+cal+";priority:med\" -c \"source ~/.bash_profile ; source ~/.bashrc ; cd ~/repos/SZ_filter ; mpirun -hostfile %hostfile% python bin/makeGrid.py "+exp+" "+gridName+" --skip-lensing \" > output"+str(time.time())+"_szgrid_"+exp+"_"+cal+".log  &"

        # do only lens
        #cmd = "nohup wq sub -r \"mode:bycore;N:"+str(numCores)+";hostfile: auto;job_name: "+exp+"_"+cal+";priority:med\" -c \"source ~/.bash_profile ; source ~/.bashrc ; cd ~/repos/SZ_filter ; mpirun -hostfile %hostfile% python bin/makeGrid.py "+exp+" "+gridName+" "+cal+" --skip-sz \" > output"+str(time.time())+"_szgrid_"+exp+"_"+cal+".log  &"


        # cmd = "nohup wq sub -r \"mode:bycore;N:"+str(numCores)+";hostfile: auto;job_name: "+exp+"_"+cal+";priority:med\" -c \"source ~/.bash_profile ; source ~/.bashrc ; cd ~/repos/SZ_filter ; mpirun -hostfile %hostfile% python bin/makeSNGrid.py "+exp+" "+cal+" \" > output"+str(time.time())+"_massgrid_"+exp+"_"+cal+".log  &"


        # cmd = "nohup wq sub -r \"mode:bycore;N:"+str(numCores)+";hostfile: auto;job_name: "+exp+"_"+cal+";priority:med\" -c \"source ~/.bash_profile ; source ~/.bashrc ; cd ~/repos/SZ_filter ; mpirun -hostfile %hostfile% python bin/makeCMBGrid.py "+exp+" "+cal+" \" > output"+str(time.time())+"_massgrid_"+exp+"_"+cal+".log  &"

        print cmd
        os.system(cmd)
        time.sleep(0.3)

