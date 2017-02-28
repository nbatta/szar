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


fishSection = "mnu-w"
#noatm = ""
noatm = "-noatm"
cal = "CMB_all"
#cal = "owl1"
derivSet = "wstep"


cosmoFisher = Config.get('fisher-'+fishSection,'saveSuffix')

paramList = Config.get('fisher-'+fishSection,'paramList').split(',')
paramLatexList = Config.get('fisher-'+fishSection,'paramLatexList').split(',')
fparams = {} 
for (key, val) in Config.items('params'):
    param = val.split(',')[0]
    fparams[key] = float(param)



"""RES STUDY"""
# cmbfisher3 = getFisher("S4-3m"+noatm,cal,cosmoFisher,paramList,derivSet)
# cmbfisher5 = getFisher("S4-5m"+noatm,cal,cosmoFisher,paramList,derivSet)
# cmbfisher6 = getFisher("S4-6m"+noatm,cal,cosmoFisher,paramList,derivSet)
# cmbfisher7 = getFisher("S4-7m"+noatm,cal,cosmoFisher,paramList,derivSet)
# fplots = FisherPlots(paramList,paramLatexList,fparams)
# fplots.addFisher('cmb3',cmbfisher3)
# fplots.addFisher('cmb5',cmbfisher5)
# fplots.addFisher('cmb6',cmbfisher6)
# fplots.addFisher('cmb7',cmbfisher7)
# fplots.plotPair(['mnu','w0'],['cmb3','cmb5','cmb6','cmb7'],labels=['S4-3m','S4-5m','S4-6m','S4-7m'],saveFile="/gpfs01/astro/www/msyriac/s4resatm_"+cal+noatm+"_"+cosmoFisher+"_"+derivSet+".png")

#"""OWL"""
# res = "5m"
# owl2 = getFisher("S4-"+res+noatm,"owl2",cosmoFisher,paramList,derivSet)
# owl1 = getFisher("S4-"+res+noatm,"owl1",cosmoFisher,paramList,derivSet)
# fplots = FisherPlots(paramList,paramLatexList,fparams)
# fplots.addFisher('S4-'+res+'-owl2',owl2)
# fplots.addFisher('S4-'+res+'-owl1',owl1)
# listFishers = ['S4-'+res+'-owl2','S4-'+res+'-owl1']
# labFishers = ['S4-'+res+' + LSST-like ($z<2$)','S4-'+res+' + LSST-like ($z<1$)']

# fplots.plotPair(['mnu','w0'],listFishers,labels=labFishers,saveFile="/gpfs01/astro/www/msyriac/s42d_mnuw0_OWL"+res+"_"+noatm+"_"+cosmoFisher+"_"+derivSet+".png")



#"""CMB OWL comparison"""
# res = "5m"
# cmbfisher5 = getFisher("S4-5m"+noatm,'CMB_all',cosmoFisher,paramList,derivSet)
# cmbfisher5pol = getFisher("S4-5m"+noatm,'CMB_pol',cosmoFisher,paramList,derivSet)
# fplots = FisherPlots(paramList,paramLatexList,fparams)
# fplots.addFisher('cmb5',cmbfisher5)
# fplots.addFisher('cmb5pol',cmbfisher5pol)
# listFishers = ['cmb5','cmb5pol']
# labFishers = ['S4-5m internal T+P','S4-5m internal P only']

# fplots.plotPair(['mnu','w0'],listFishers,labels=labFishers,saveFile="/gpfs01/astro/www/msyriac/s42d_mnuw0_cmbTP"+res+"_"+noatm+"_"+cosmoFisher+"_"+derivSet+".png")


#"""CMB OWL comparison"""

res = "5m"
owl1x = getFisher("S4-5m"+noatm,'owl2',cosmoFisher,paramList,"wstep")
owl2x = getFisher("S4-5m"+noatm,'owl2',cosmoFisher,paramList,"wstep2x")
fplots = FisherPlots(paramList,paramLatexList,fparams)
fplots.addFisher('owl1x',owl1x)
fplots.addFisher('owl2x',owl2x)
listFishers = ['owl1x','owl2x']
labFishers = ['S4-5m LSST-like','S4-5m LSST-like 2x error']

fplots.plotPair(['mnu','w0'],listFishers,labels=labFishers,saveFile="/gpfs01/astro/www/msyriac/s42d_mnuw0_2x"+res+"_"+noatm+"_"+cosmoFisher+"_"+derivSet+".png")
