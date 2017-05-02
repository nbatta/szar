import time
import os

numParams = 15
numCores = 2*numParams+1

#expList = ['S4-1.0-0.4','S4-1.5-0.4','S4-1.5-0.7','S4-1.5-0.3','S4-1.5-0.2','S4-1.5-0.1','S4-1.5-0.05','S4-2.0-0.4','S4-2.5-0.4','S4-3.0-0.4']
#expList = ['SO-v2-6m'] #'S4-1.5-0.3','S4-1.5-0.2','S4-1.5-0.1']
expList = ['SO-v2-6m','SO-v2']
#expList = ['SO-v2-6m','SO-v2','SO-v2-6m-noatm','SO-v2-noatm'] #'S4-1.5-#expList = ['S4-1.0-0.4','S4-2.0-0.4','S4-3.0-0.4']
#calList = ['CMB_all']
#expList = ['SO-v2'] #'S4-1.5-0.3','S4-1.5-0.2','S4-1.5-0.1']
#expList = ['S4-1.5-0.7','S4-1.5-0.3','S4-1.5-0.2','S4-1.5-0.1','S4-1.5-0.05']
calList = ['CMB_all']#,'CMB_pol']#,'CMB_all_miscentered','CMB_pol_miscentered']

gridName = "grid-default"

from ConfigParser import SafeConfigParser 
iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)
bigDataDir = Config.get('general','bigDataDirectory')
version = Config.get('general','version')



for exp in expList:

    for cal in calList:


        massGridName = bigDataDir+"lensgrid_"+exp+"_"+gridName+"_"+cal+ "_v" + version+".pkl"

        #cmd = " mpirun -np " +str(numCores) + " python bin/makeDerivs.py allParams "+exp+" "+cal+" "+massGridName+"  > output"+str(time.time())+".log  2>&1 &"

        # all on astro
        cmd = "nohup wq sub -r \"mode:bycore;N:"+str(numCores)+";hostfile: auto;job_name: "+exp+"_"+cal+";priority:med\" -c \"source ~/.bash_profile ; source ~/.bashrc ; cd ~/repos/szar ; mpirun -hostfile %hostfile% python bin/makeDerivs.py allParams "+exp+" "+gridName+" "+cal+" "+massGridName+" \" > output"+str(time.time())+"_cmbderiv_"+exp+"_"+cal+".log  &"

        # cmd = "nohup wq sub -r \"mode:bycore;N:"+str(numCores)+";hostfile: auto;job_name: "+exp+"_"+cal+";priority:med\" -c \"source ~/.bash_profile ; source ~/.bashrc ; cd ~/repos/szar ; mpirun -hostfile %hostfile% python bin/makeDerivs.py b_wl "+exp+" "+gridName+" "+cal+" "+massGridName+" \" > output"+str(time.time())+"_cmbderiv_"+exp+"_"+cal+".log  &"

        # w0 on astro
        #cmd = "nohup wq sub -r \"mode:bycore;N:"+str(numCores)+";hostfile: auto;job_name: "+exp+"_"+cal+";priority:med\" -c \"source ~/.bash_profile ; source ~/.bashrc ; cd ~/repos/szar ; mpirun -hostfile %hostfile% python bin/makeDerivs.py w0 "+exp+" "+gridName+" "+cal+" "+massGridName+" \" > output"+str(time.time())+"_cmbderiv_"+exp+"_"+cal+".log  &"
        
        print cmd
        os.system(cmd)
        time.sleep(0.3)

