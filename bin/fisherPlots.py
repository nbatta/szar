import matplotlib
matplotlib.use('Agg')
from ConfigParser import SafeConfigParser 
import cPickle as pickle
import sys

from orphics.tools.io import FisherPlots


def getFisher(expName,calName,saveName,inParamList,suffix):
    saveId = expName + "_" + calName + "_" + suffix

    paramList,FisherTot = pickle.load(open("data/savedFisher_"+saveId+"_"+saveName+".pkl",'rb'))
    assert paramList==inParamList
    return FisherTot




iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)


paramList = Config.get('fisher','paramList').split(',')
paramLatexList = Config.get('fisher','paramLatexList').split(',')
fparams = {} 
for (key, val) in Config.items('params'):
    param = val.split(',')[0]
    fparams[key] = float(param)


#noatm = ""
noatm = "-noatm"

#cal = "CMB_all"
cal = "owl2"

cosmoFisher = "planck_mwcdm"
derivSet = "wstep"

cmbfisher3 = getFisher("S4-3m"+noatm,cal,cosmoFisher,paramList,derivSet)
cmbfisher5 = getFisher("S4-5m"+noatm,cal,cosmoFisher,paramList,derivSet)
cmbfisher6 = getFisher("S4-6m"+noatm,cal,cosmoFisher,paramList,derivSet)
cmbfisher7 = getFisher("S4-7m"+noatm,cal,cosmoFisher,paramList,derivSet)


fplots = FisherPlots(paramList,paramLatexList,fparams)
fplots.addFisher('cmb3',cmbfisher3)
fplots.addFisher('cmb5',cmbfisher5)
fplots.addFisher('cmb6',cmbfisher6)
fplots.addFisher('cmb7',cmbfisher7)
fplots.plotPair(['mnu','w0'],['cmb3','cmb5','cmb6','cmb7'],labels=['S4-3m','S4-5m','S4-6m','S4-7m'],saveFile="/gpfs01/astro/www/msyriac/s4resatm_"+cal+noatm+"_"+cosmoFisher+"_"+derivSet+".png")
