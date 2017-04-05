import os
cmdRoot = "python bin/calcS8Fisher.py "

expList = ['S4','SO']
resList = ['-3m','-5m','-6m','-7m']
atmList = ['']#,'-noatm']
#calList = ['CMB_all','owl2']#,'CMB_pol','owl1']
calList = ['CMB_pol']#,'CMB_pol','owl1']
#calList = ['owl1']#,'CMB_pol','owl1']

gridList = ['grid-default']


for exp in expList:
    for res in resList:
        for atm in atmList:
            for cal in calList:
                for grid in gridList:
            
                    cmd = cmdRoot + exp+res+atm+" "+grid+" "+cal+" "
                    print cmd
                    os.system(cmd)
