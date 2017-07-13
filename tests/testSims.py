import szar.sims as s
from orphics.tools.io import Plotter,dictFromSection,listFromConfig
from ConfigParser import SafeConfigParser 

iniFile = "input/params.ini"
#iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)
constDict = dictFromSection(Config,'constants')

sim = s.BattagliaSims(constDict)
sim.mapReader(plotRel=True)

print "done"



