import time
import os

numParams = 1
numCores = 2*numParams+1
#expList = ['SO-v2-6m'] #'S4-1.5-0.3','S4-1.5-0.2','S4-1.5-0.1']
expList = ['SO-v2-6m','SO-v2']

#expList = ['SO-v2-6m','SO-v2','SO-v2-6m-noatm','SO-v2-noatm'] #'S4-1.5-#expList = ['S4-1.0-0.4','S4-1.5-0.4','S4-1.5-0.7','S4-1.5-0.3','S4-1.5-0.2','S4-1.5-0.1','S4-1.5-0.05','S4-2.0-0.4','S4-2.5-0.4','S4-3.0-0.4','SO-v2','SO-v2-6m']
#expList = ['SO-5m','SO-6m','SO-7m','S4-5m','S4-6m','S4-7m']
#calList = ['CMB_all','CMB_pol','CMB_all_miscentered','CMB_pol_miscentered']
calList = ['owl2']#,'CMB_pol']
#calList = ['testCal']


from configparser import SafeConfigParser 
iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)
bigDataDir = Config.get('general','bigDataDirectory')
version = Config.get('general','version')

for exp in expList:

    for cal in calList:
        grid = "grid-"+cal

        massGridName = bigDataDir+"lensgrid_grid-"+cal+"_"+cal+".pkl"

        #massGridName = bigDataDir+"lensgrid_"+exp+"_"+grid+"_"+cal+ "_v" + version+".pkl"

        #cmd = " mpirun -np " +str(numCores) + " python bin/makeWaDeriv.py "+exp+" "+cal+" "+massGridName+" data/forDerivsStep0.1 0.1 > output"+str(time.time())+".log  2>&1 &"

        cmd = "nohup wq sub -r \"mode:bycore;N:"+str(numCores)+";hostfile: auto;job_name: ohhaithere;priority:med\" -c \"source ~/.bash_profile ; source ~/.bashrc ; cd ~/repos/szar ; mpirun -hostfile %hostfile% python bin/makeWaDeriv.py "+exp+" "+grid+" "+cal+" "+massGridName+" \" > output"+str(time.time())+".log  &"
        #cmd = "nohup wq sub -r \"mode:bycore;N:"+str(numCores)+";hostfile: auto;job_name: ohhaithere;priority:med\" -c \"source ~/.bash_profile ; source ~/.bashrc ; cd ~/repos/szar ; mpirun -hostfile %hostfile% python bin/makeDerivs.py w0 "+exp+" "+cal+" "+massGridName+" \" > output"+str(time.time())+".log  &"
        
        print(cmd)
        os.system(cmd)
        time.sleep(0.3)

