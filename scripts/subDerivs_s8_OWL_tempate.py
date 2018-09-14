from __future__ import print_function
from orphics.tools.io import listFromConfig
import numpy as np
import time
import os

expList = ['SO-3m','SO-5m','SO-6m','SO-7m','S4-3m','S4-5m','S4-6m','S4-7m','S4-5m-noatm','S4-6m-noatm','S4-7m-noatm','SO-5m-noatm','SO-6m-noatm','SO-7m-noatm','SO-3m-noatm','S4-3m-noatm']

calList = ['owl1','owl2']



from configparser import SafeConfigParser 
iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)
bigDataDir = Config.get('general','bigDataDirectory')
version = Config.get('general','version')




for exp in expList:

    for cal in calList:

        gridName = "grid-"+cal

        zs = listFromConfig(Config,gridName,'zrange')
        z_edges = np.arange(zs[0],zs[1]+zs[2],zs[2])

        numCores = z_edges.size
        
        massGridName = bigDataDir+"lensgrid_"+gridName+"_"+cal+".pkl"

        cmd = "nohup wq sub -r \"mode:bycore;N:"+str(numCores)+";hostfile: auto;job_name: "+exp+"_"+cal+";priority:med\" -c \"source ~/.bash_profile ; source ~/.bashrc ; cd ~/repos/szar ; mpirun -hostfile %hostfile% python bin/makeS8Derivs.py "+exp+" "+gridName+" "+cal+" "+massGridName+" \" > output"+str(time.time())+"_cmbderiv_"+exp+"_"+cal+".log  &"
        
        print(cmd)
        os.system(cmd)
        time.sleep(0.3)

