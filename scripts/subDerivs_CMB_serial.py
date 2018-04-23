import time
import os
from configparser import SafeConfigParser 


#expList = ['S4-3m-noatm','S4-5m-noatm','S4-6m-noatm','S4-7m-noatm','SO-3m-noatm','SO-5m-noatm','SO-6m-noatm','SO-7m-noatm']#,'SO-6m','SO-7m','S4-5m','S4-6m','S4-7m']
#expList = ['S4-5m','S4-7m']
#expList = ['SO-3m','SO-5m','SO-6m','SO-7m','S4-3m','S4-5m','S4-6m','S4-7m','S4-5m-noatm','S4-6m-noatm','S4-7m-noatm','SO-5m-noatm','SO-6m-noatm','SO-7m-noatm','SO-3m-noatm','S4-3m-noatm']
#expList = ['CCATP']
#expList = ['SO-v2A','SO-v2A-6m','SO-v2B','SO-v2B-6m','SO-v2D','SO-v2D-6m','SO-v2E','SO-v2E-6m']
#expList = ['AdvAct-s16_ud','AdvAct-s16_dp','AdvAct-s16_md','AdvAct-s16_wd','AdvAct-s16_uw']
#expList = ['SO-v2C','SO-v2C-6m']
#expList = ['S4-1.0-CDT-max']#,'S4-1.5-CDT']
#expList = ['SO-v3-base-40','SO-v3-goal-40','SO-v3-base-20']
expList = ['CMB-Probe-v3-1']
#expList = ['AdvACT_S19']
#expList = ['SO-v3-goal-20','SO-v3-base-10','SO-v3-goal-10']

#'SO-5m-noatm','SO-6m-noatm','SO-7m-noatm',
#expList = ['AdvAct']
calList = ['CMB_all_PICO']#,'owl1']#,'owl1']

gridName = "grid-default"

iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)
bigDataDir = Config.get('general','bigDataDirectory')
version = Config.get('general','version')

manualParamList = Config.get('general','manualParams').split(',')

paramList = [] # the parameters that can be varied                                                                                                                                   
fparams = {}   # the                                                                                                                                                                 
stepSizes = {}
for (key, val) in Config.items('params'):
    if key in manualParamList: continue
    if ',' in val:
        param, step = val.split(',')
        paramList.append(key)
        fparams[key] = float(param)
        stepSizes[key] = float(step)
    else:
        fparams[key] = float(val)
                    
for exp in expList:

    for cal in calList:

        massGridName = bigDataDir+"lensgrid_"+exp+"_"+gridName+"_"+cal+ "_v" + version+".pkl"
        #massGridName = bigDataDir+"lensgrid_"+exp+"_grid-"+cal+"_"+cal+".pkl"

        #cmd = "mpirun -np "+str(numCores)+" python bin/makeDerivs.py allParams "+exp+" "+gridName+" "+cal+" "+massGridName+" > output"+str(time.time())+"_cmbderiv_"+exp+"_"+cal+".log  &"
        for paramsNow in paramList:
            cmd = "mpirun -np 3 python bin/makeDerivs.py "+paramsNow+" "+exp+" "+gridName+" "+cal+" "+massGridName

            print(cmd)
            os.system(cmd)
            time.sleep(0.3)

