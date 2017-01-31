
import orphics.tools.io as io
from ConfigParser import SafeConfigParser 
import sys


paramName = sys.argv[1]
key = paramName
print "Calculating derivative for ", key

iniFile = "input/params.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)

cosmologyName = 'LACosmology' # from ini file
cosmoListDict = io.dictOfListsFromSection(Config,cosmologyName)

experimentName = ""

saveId = experimentName + cosmologyName




upDict = cosmoDict.copy()
dnDict = cosmoDict.copy()

try:
    upDict[key] = cosmoListDect[key][0] + cosmoListDect[key][1]
    dnDict[key] = cosmoListDect[key][0] - cosmoListDect[key][1]
except:
    print "No step size specified for ", key
    sys.exit()    
    


ccUp = ClusterCosmology(upDict,constDict,lmax)
ccDn = ClusterCosmology(dnDict,constDict,lmax)


beam = listFromConfig(Config,experimentName,'beams')
noise = listFromConfig(Config,experimentName,'noises')
freq = listFromConfig(Config,experimentName,'freqs')
lmax = int(Config.getfloat(experimentName,'lmax'))
lknee = Config.getfloat(experimentName,'lknee')
alpha = Config.getfloat(experimentName,'alpha')
fsky = Config.getfloat(experimentName,'fsky')


Nmup = getNmzq(ccUp,experiment)
Nmdn = getNmzq(ccDn,experiment)


np.save("data/"+saveId+"_"+key+"_up.dat",Nmup)
np.save("data/"+saveId+"_"+key+"_dn.dat",Nmdn)

               
