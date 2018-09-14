from __future__ import print_function
from orphics.tools.io import listFromConfig
import numpy as np
import time
import os

#expList = ['S4-1.0-0.4','S4-1.5-0.4','S4-1.5-0.7','S4-1.5-0.3','S4-1.5-0.2','S4-1.5-0.1','S4-1.5-0.05','S4-2.0-0.4','S4-2.5-0.4','S4-3.0-0.4']
#expList = ['S4-1.5-0.3','S4-1.5-0.2','S4-1.5-0.1']
expList = ['SO-v2'] #'S4-1.5-0.3','S4-1.5-0.2','S4-1.5-0.1']
calList = ['CMB_all','CMB_pol','CMB_all_miscentered','CMB_pol_miscentered']


gridName = "grid-default"

from configparser import SafeConfigParser 
iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)
bigDataDir = Config.get('general','bigDataDirectory')
version = Config.get('general','version')


zs = listFromConfig(Config,gridName,'zrange')
z_edges = np.arange(zs[0],zs[1]+zs[2],zs[2])

numCores = z_edges.size


for exp in expList:

    for cal in calList:


        massGridName = bigDataDir+"lensgrid_"+exp+"_"+gridName+"_"+cal+ "_v" + version+".pkl"


        cmd = "nohup wq sub -r \"mode:bycore;N:"+str(numCores)+";hostfile: auto;job_name: "+exp+"_"+cal+";priority:med\" -c \"source ~/.bash_profile ; source ~/.bashrc ; cd ~/repos/szar ; mpirun -hostfile %hostfile% python bin/makeS8Derivs.py "+exp+" "+gridName+" "+cal+" "+massGridName+" \" > output"+str(time.time())+"_cmbderiv_"+exp+"_"+cal+".log  &"
        
        print(cmd)
        os.system(cmd)
        time.sleep(0.3)

