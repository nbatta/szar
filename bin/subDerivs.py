import time
import os

numParams = 14
#numParams = 1
numCores = 2*numParams+1
#expList = ['S4-3m-noatm','S4-5m-noatm','S4-6m-noatm','S4-7m-noatm']#,'SO-6m-noatm','SO-7m-noatm']
#expList = ['SO-3m','SO-5m','SO-6m','SO-7m','S4-3m','S4-5m','S4-6m','S4-7m']
#calList = ['CMB_all_coarse','CMB_pol_coarse']

expList = ['S4-3m-noatm','S4-5m-noatm','S4-6m-noatm','S4-7m-noatm']
calList = ['CMB_all_coarse']


#calList = ['CMB_all','CMB_pol']#,'CMB_all_nodelens','CMB_pol_nodelens']#,'OWL1','OWL2','CMBAllOWL1','CMBAllOWL2','CMBPoexpList = ['SO-5m','SO-6m','SO-7m','S4-5m','S4-6m','S4-7m']
#calList = ['CMB_all','CMB_pol','CMB_all_nodelens','CMB_pol_nodelens']#,'OWL1','OWL2','CMBAllOWL1','CMBAllOWL2','CMBPolOWL1','CMBPolOWL2']
# expList = ['SO-5m']
# calList = ['testGrid']


from ConfigParser import SafeConfigParser 
iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)
bigDataDir = Config.get('general','bigDataDirectory')

for exp in expList:

    for cal in calList:


        massGridName = bigDataDir+exp+cal+".pkl"

        #cmd = " mpirun -np " +str(numCores) + " python bin/makeDerivs.py allParams "+exp+" "+cal+" "+massGridName+"  > output"+str(time.time())+".log  2>&1 &"

        cmd = "nohup wq sub -r \"mode:bycore;N:"+str(numCores)+";hostfile: auto;job_name: "+exp+"_"+cal+";priority:med\" -c \"source ~/.bash_profile ; source ~/.bashrc ; cd ~/repos/SZ_filter ; mpirun -hostfile %hostfile% python bin/makeDerivs.py allParams "+exp+" "+cal+" "+massGridName+" \" > output"+str(time.time())+"_cmbderiv_"+exp+"_"+cal+".log  &"
        #cmd = "nohup wq sub -r \"mode:bycore;N:"+str(numCores)+";hostfile: auto;job_name: ohhaithere;priority:med\" -c \"source ~/.bash_profile ; source ~/.bashrc ; cd ~/repos/SZ_filter ; mpirun -hostfile %hostfile% python bin/makeDerivs.py w0 "+exp+" "+cal+" "+massGridName+" \" > output"+str(time.time())+".log  &"
        
        print cmd
        os.system(cmd)
        time.sleep(0.3)

