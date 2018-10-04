from __future__ import print_function
import time
import os

#expList = ['SO-v2-6m'] #'S4-1.5-0.3','S4-1.5-0.2','S4-1.5-0.1']
#expList = ['S4-1.0-CDT-max']#,'S4-1.5-CDT']
#expList = ['AdvACT_S19']
#expList = ['CMB-Probe-v3-1']
expList = ['SO-v3-goal-40']#,'SO-v3-base-40','SO-v3-goal-20','SO-v3-base-20','SO-v3-goal-10','SO-v3-base-10']
#expList = ['SO-v2-6m','SO-v2','SO-v2-6m-noatm','SO-v2-noatm'] #'S4-1.5-#expList = ['S4-1.0-0.4','S4-1.5-0.4','S4-1.5-0.7','S4-1.5-0.3','S4-1.5-0.2','S4-1.5-0.1','S4-1.5-0.05','S4-2.0-0.4','S4-2.5-0.4','S4-3.0-0.4','SO-v2','SO-v2-6m']
#expList = ['SO-5m','SO-6m','SO-7m','S4-5m','S4-6m','S4-7m']
calList = ['CMB_all']#,'CMB_pol','CMB_all_miscentered','CMB_pol_miscentered']
#calList = ['CMB_all_PICO']#,'CMB_pol']
#calList = ['testCal']

grid = "grid-default"

from configparser import SafeConfigParser 
iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)
bigDataDir = Config.get('general','bigDataDirectory')
version = Config.get('general','version')

for exp in expList:

    for cal in calList:

        massGridName = bigDataDir+"lensgrid_"+exp+"_"+grid+"_"+cal+ "_v" + version+".pkl"

        cmd = "mpirun -np 3 python bin/makeWaDeriv.py "+exp+" "+grid+" "+cal+" "+massGridName
        
        print(cmd)
        os.system(cmd)
        time.sleep(0.3)

