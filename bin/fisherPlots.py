import matplotlib
matplotlib.use('Agg')
from ConfigParser import SafeConfigParser 
import cPickle as pickle
import sys

from orphics.tools.io import FisherPlots


def getFisher(expName,calName,saveName,inParamList):
    saveId = expName + "_" + calName + "_" + suffix

    paramList,FisherTot = pickle.load(open("data/savedFisher_"+saveId+"_"+saveName+".pkl",'rb'))
    assert paramList==inParamList
    return FisherTot




iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)

suffix = Config.get('general','suffix')

paramList = Config.get('fisher','paramList').split(',')
paramLatexList = Config.get('fisher','paramLatexList').split(',')
fparams = {} 
for (key, val) in Config.items('params'):
    param = val.split(',')[0]
    fparams[key] = float(param)



owlfisher = getFisher("SO-6m","owl1","planck_mwcdm",paramList)
cmbfisher = getFisher("SO-6m","CMB_all","planck_mwcdm",paramList)


fplots = FisherPlots(paramList,paramLatexList,fparams)
fplots.addFisher('owl',owlfisher)
fplots.addFisher('cmb',cmbfisher)
#fplots.trianglePlot(['H0','mnu','w'],['planckS4OWLClusters','planckS4CMBClusters'],['-','--'])
fplots.plotPair(['mnu','w0'],['owl','cmb'],cols=['red','blue'],lss=['-','--'],labels=['owl','cmb'],saveFile="output/test.png")
