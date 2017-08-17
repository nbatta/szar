import os
cmdRoot = "python bin/calcFisher.py "

#expList = ['S4-1.0-0.4','S4-1.5-0.4','S4-1.5-0.7','S4-1.5-0.3','S4-1.5-0.2','S4-1.5-0.1','S4-1.5-0.05','S4-2.0-0.4','S4-2.5-0.4','S4-3.0-0.4']
#calList = ['CMB_all','CMB_pol','CMB_all_miscentered','CMB_pol_miscentered']

expList = ['S4-1.0-0.4','S4-1.5-0.4','S4-2.0-0.4','S4-2.5-0.4','S4-3.0-0.4']
calList = ['CMB_all']#,'CMB_pol','owl1','owl2']

#expList = ['S4','SO']
#expList = ['S4']
resList = ['']#-3m','-5m','-6m','-7m']
#atmList = ['-noatm']
atmList = ['']#,'-noatm']
#atmList = ['-noatm']
#calList = ['CMB_all','CMB_pol','CMB_all_nodelens','CMB_pol_nodelens']
#calList = ['CMB_all','CMB_pol','owl1','owl2']
#calList = ['CMB_all','owl2']#,'CMB_pol','owl1']
#calList = ['CMB_pol']#,'CMB_pol','owl1']
#calList = ['owl1']#,'CMB_pol','owl1']
fishList = ['mnu-w0-wa']#,'mnu','w0']#,'w','mnu-w']#,'mnu-cvltau','w-cvltau','mnu-w-cvltau','mnu-notau','w-notau','mnu-w-notau']


for exp in expList:
    for res in resList:
        for atm in atmList:
            for cal in calList:
                for fisher in fishList:
                    if cal[:3]=="owl":
                        grid = "grid-"+cal
                    else:
                        grid = "grid-default"

                    cmd = cmdRoot + exp+res+atm+" "+grid+" "+cal+" "+fisher
                    print cmd
                    os.system(cmd)
