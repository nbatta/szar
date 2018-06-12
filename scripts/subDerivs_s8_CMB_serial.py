from orphics.tools.io import listFromConfig
import numpy as np
import time
import os

#expList = ['S4-1.0-0.4','S4-1.5-0.4','S4-1.5-0.7','S4-1.5-0.3','S4-1.5-0.2','S4-1.5-0.1','S4-1.5-0.05','S4-2.0-0.4','S4-2.5-0.4','S4-3.0-0.4']
#expList = ['S4-1.5-0.3','S4-1.5-0.2','S4-1.5-0.1']
#expList = ['S4-1.0-CDT-max']#,'S4-1.5-CDT'] #'S4-1.5-0.3','S4-1.5-0.2','S4-1.5-0.1']
#expList = ['SO-v3-base-40','SO-v3-goal-40','SO-v3-base-20','SO-v3-goal-20','SO-v3-base-10']
expList = ['SO-v3-goal-40']#,'SO-v3-goal-10']
#expList = ['AdvACT_S19']
#expList = ['SO-v3-goal-20','SO-v3-base-10','SO-v3-goal-10']                                                              
calList = ['CMB_all']#,'CMB_pol','CMB_all_miscentered','CMB_pol_miscentered']


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

numCores = 3


for exp in expList:

    for cal in calList:

        massGridName = bigDataDir+"lensgrid_"+exp+"_"+gridName+"_"+cal+ "_v" + version+".pkl"

        cmd = "python bin/makeS8Derivs_serial.py "+exp+" "+gridName+" "+cal+" "+massGridName
        
        print(cmd)
        os.system(cmd)
        time.sleep(0.3)

