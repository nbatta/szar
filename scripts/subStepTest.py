import numpy as np
import time
import sys,os
from ConfigParser import SafeConfigParser 
import cPickle as pickle

from orphics.tools.io import dictFromSection, listFromConfig,dictOfListsFromSection, Plotter

iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)

stepdict = dictOfListsFromSection(Config,'steps')

exp = "S4-1.0-0.4"
grid = "grid-default"
cal = "CMB_all"

for key in stepdict.keys():

    cmd = "quick_wq.sh python tests/testSteps.py "+exp+" "+grid+" "+cal+ " " +key 

    print cmd
    os.system(cmd)
    time.sleep(0.3)

