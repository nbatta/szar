from __future__ import print_function
import numpy as np
import time
import sys,os
from configparser import SafeConfigParser 
import pickle as pickle

from orphics.tools.io import dictFromSection, listFromConfig,dictOfListsFromSection, Plotter

iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)

stepdict = dictOfListsFromSection(Config,'steps')

exp = "S4-1.0-paper"
grid = "grid-default"
cal = "CMB_all"

# grid = "grid-owl2"
# cal = "owl2"

for key in ['b_wl']: #stepdict.keys():

    cmd = "quick_wq gen6 python tests/testSteps.py "+exp+" "+grid+" "+cal+ " " +key 

    print(cmd)
    os.system(cmd)
    time.sleep(0.3)

