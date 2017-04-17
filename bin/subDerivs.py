import time
import os

numParams = 15
#numParams = 1
numCores = 2*numParams+1
#expList = ['S4-3m-noatm','S4-5m-noatm','S4-6m-noatm','S4-7m-noatm']#,'SO-6m-noatm','SO-7m-noatm']
#expList = ['SO-3m','SO-5m','SO-6m','SO-7m','S4-3m','S4-5m','S4-6m','S4-7m']
#calList = ['CMB_all_coarse','CMB_pol_coarse']


expList = ['S4-3.0-0.4']
calList = ['CMB_pol_miscentered']


#expList = ['S4-7m']
#calList = ['CMB_all']
#expList = ['SO-3m','SO-5m','SO-6m','SO-7m','S4-3m','S4-5m','S4-6m','S4-7m','S4-5m-noatm','S4-6m-noatm','S4-7m-noatm','SO-5m-noatm','SO-6m-noatm','SO-7m-noatm','SO-3m-noatm','S4-3m-noatm']
#calList = ['CMB_all','CMB_pol','CMB_all_miscentered','CMB_pol_miscentered']
#calList = ['CMB_all','CMB_pol','CMB_all_nodelens','CMB_pol_nodelens']
#expList = ['SO-3m','SO-5m','SO-6m','SO-7m']
#expList = ['S4-3m','S4-5m','S4-6m','S4-7m']
#expList = ['S4-1.5arc-noatm']
#calList = ['CMB_pol','CMB_all']

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
        cmd = "nohup wq sub -r \"mode:bycore;N:"+str(numCores)+";hostfile: auto;job_name: "+exp+"_"+cal+";priority:med\" -c \"source ~/.bash_profile ; source ~/.bashrc ; cd ~/repos/SZ_filter ; mpirun -hostfile %hostfile% python bin/makeDerivs.py allParams "+exp+" "+gridName+" "+cal+" "+massGridName+" \" > output"+str(time.time())+"_cmbderiv_"+exp+"_"+cal+".log  &"

        # cmd = "nohup wq sub -r \"mode:bycore;N:"+str(numCores)+";hostfile: auto;job_name: "+exp+"_"+cal+";priority:med\" -c \"source ~/.bash_profile ; source ~/.bashrc ; cd ~/repos/SZ_filter ; mpirun -hostfile %hostfile% python bin/makeDerivs.py b_wl "+exp+" "+gridName+" "+cal+" "+massGridName+" \" > output"+str(time.time())+"_cmbderiv_"+exp+"_"+cal+".log  &"

        # w0 on astro
        #cmd = "nohup wq sub -r \"mode:bycore;N:"+str(numCores)+";hostfile: auto;job_name: "+exp+"_"+cal+";priority:med\" -c \"source ~/.bash_profile ; source ~/.bashrc ; cd ~/repos/SZ_filter ; mpirun -hostfile %hostfile% python bin/makeDerivs.py w0 "+exp+" "+gridName+" "+cal+" "+massGridName+" \" > output"+str(time.time())+"_cmbderiv_"+exp+"_"+cal+".log  &"
        
        print cmd
        os.system(cmd)
        time.sleep(0.3)

