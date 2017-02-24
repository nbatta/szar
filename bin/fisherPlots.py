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



#owlfisher1 = getFisher("SO-7m","owl1","planck_mwcdm",paramList)
#owlfisher52 = getFisher("SO-5m","owl2","planck_mwcdm",paramList)
#cmbfisher5cvl = getFisher("SO-5m","CMB_all","planck_mwcdm_cvltau",paramList)
cmbfisher5 = getFisher("SO-5m-noatm","CMB_all","planck_mwcdm",paramList)
cmbfisher6 = getFisher("SO-6m-noatm","CMB_all","planck_mwcdm",paramList)
cmbfisher7 = getFisher("SO-7m-noatm","CMB_all","planck_mwcdm",paramList)
#cmbfisher47 = getFisher("S4-7m","CMB_all","planck_mwcdm",paramList)
#cmbfisher7so = getFisher("SO-7m","CMB_all","planck_mwcdm",paramList)


fplots = FisherPlots(paramList,paramLatexList,fparams)
#fplots.addFisher('owl52',owlfisher52)
fplots.addFisher('cmb5',cmbfisher5)
fplots.addFisher('cmb6',cmbfisher6)
fplots.addFisher('cmb7',cmbfisher7)
#fplots.addFisher('cmb47',cmbfisher47)
#fplots.addFisher('cmb7so',cmbfisher7so)
#fplots.trianglePlot(['H0','mnu','w'],['planckS4OWLClusters','planckS4CMBClusters'],['-','--'])
#fplots.plotPair(['mnu','w0'],['cmb5','cmb6','cmb7'],labels=['SO-5m','SO-6m','SO-7m'],saveFile="output/test.png")
#fplots.plotPair(['mnu','w0'],['cmb5','cmb7'],labels=['S4-5m','S4-7m'],saveFile="output/test.png")
#fplots.plotPair(['mnu','w0'],['owl2','cmb7'],labels=['LSST z<2','S4-7m'],saveFile="output/owlcmb2.png")
#fplots.plotPair(['mnu','w0'],['owl52','cmb5'],labels=['SO-5m + LSST','SO-5m + internal \n (CMB halo lensing)'],saveFile="output/soint.png")
#fplots.plotPair(['mnu','w0'],['cmb5','cmb6','cmb7'],labels=['SO-5m internal','SO-6m internal','SO-7m internal'],saveFile="output/sores_cvltau.png")
fplots.plotPair(['mnu','w0'],['cmb5','cmb6','cmb7'],labels=['SO-5m no-atm','SO-6m noatm', 'SO-7m noatm'],saveFile="output/socomp.png")
