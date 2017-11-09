import time
import os

#expList = ['S4-1.0-0.4','S4-1.5-0.4','S4-1.5-0.7','S4-1.5-0.3','S4-1.5-0.2','S4-1.5-0.1','S4-1.5-0.05','S4-2.0-0.4','S4-2.5-0.4','S4-3.0-0.4','SO-v2','SO-v2-6m']

#expList = ['S4-1.0-0.4','S4-2.0-0.4','S4-3.0-0.4']
#calList = ['CMB_all']

#,'CMB_pol']#,'CMB_all_miscentered','CMB_pol_miscentered']
expList = ['SO-v2-6m']
#expList = ['SO-v2-6m','SO-v2','SO-v2-6m-noatm','SO-v2-noatm'] #'S4-1.5-0.3','S4-1.5-0.2','S4-1.5-0.1']
calList = ['CMB_all']#,'CMB_pol']#,'CMB_all_miscentered','CMB_pol_miscentered']

# expList = ['S4-3.0-0.4']
#calList = ['CMB_pol_miscentered']

numCores = 72

gridName = "grid-default"


for exp in expList:
    for cal in calList:

        #do both only gen4
        cmd = "nohup wq sub -r \"mode:bycore;N:"+str(numCores)+";hostfile: auto;job_name: "+exp+"_"+cal+";group:[gen4];priority:med\" -c \"source ~/.bash_profile ; source ~/.bashrc ; cd ~/repos/szar ; mpirun -hostfile %hostfile% python bin/makeGrid.py "+exp+" "+gridName+" "+cal+"  \" > output"+str(time.time())+"_szgrid_"+exp+"_"+cal+".log  &"

        
        # do both
        #cmd = "nohup wq sub -r \"mode:bycore;N:"+str(numCores)+";hostfile: auto;job_name: "+exp+"_"+cal+";priority:med\" -c \"source ~/.bash_profile ; source ~/.bashrc ; cd ~/repos/szar ; mpirun -hostfile %hostfile% python bin/makeGrid.py "+exp+" "+gridName+" "+cal+"  \" > output"+str(time.time())+"_szgrid_"+exp+"_"+cal+".log  &"

        
        # do only sz
        #cmd = "nohup wq sub -r \"mode:bycore;N:"+str(numCores)+";hostfile: auto;job_name: "+exp+"_"+cal+";priority:med\" -c \"source ~/.bash_profile ; source ~/.bashrc ; cd ~/repos/szar ; mpirun -hostfile %hostfile% python bin/makeGrid.py "+exp+" "+gridName+" --skip-lensing \" > output"+str(time.time())+"_szgrid_"+exp+"_"+cal+".log  &"

        # do only lens
        #cmd = "nohup wq sub -r \"mode:bycore;N:"+str(numCores)+";hostfile: auto;job_name: "+exp+"_"+cal+";priority:med\" -c \"source ~/.bash_profile ; source ~/.bashrc ; cd ~/repos/szar ; mpirun -hostfile %hostfile% python bin/makeGrid.py "+exp+" "+gridName+" "+cal+" --skip-sz \" > output"+str(time.time())+"_szgrid_"+exp+"_"+cal+".log  &"


        # cmd = "nohup wq sub -r \"mode:bycore;N:"+str(numCores)+";hostfile: auto;job_name: "+exp+"_"+cal+";priority:med\" -c \"source ~/.bash_profile ; source ~/.bashrc ; cd ~/repos/szar ; mpirun -hostfile %hostfile% python bin/makeSNGrid.py "+exp+" "+cal+" \" > output"+str(time.time())+"_massgrid_"+exp+"_"+cal+".log  &"


        # cmd = "nohup wq sub -r \"mode:bycore;N:"+str(numCores)+";hostfile: auto;job_name: "+exp+"_"+cal+";priority:med\" -c \"source ~/.bash_profile ; source ~/.bashrc ; cd ~/repos/szar ; mpirun -hostfile %hostfile% python bin/makeCMBGrid.py "+exp+" "+cal+" \" > output"+str(time.time())+"_massgrid_"+exp+"_"+cal+".log  &"

        print(cmd)
        os.system(cmd)
        time.sleep(0.3)

