import matplotlib
matplotlib.use('Agg')
from ConfigParser import SafeConfigParser 
import cPickle as pickle
import sys, os
import numpy as np

from orphics.tools.io import FisherPlots


def getFisher(expName,gridName,calName,saveName,inParamList,suffix):
    saveId = expName + "_" + gridName+ "_" + calName + "_" + suffix

    paramList,FisherTot = pickle.load(open(bigDataDir+"savedFisher_"+saveId+"_"+saveName+".pkl",'rb'))
    assert paramList==inParamList
    return FisherTot


out_dir = os.environ['WWW']+"paper/"

iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)
bigDataDir = Config.get('general','bigDataDirectory')


fishSection = "mnu-w0-wa"
#fishSection = "lcdm"

noatm = ""
cal = "CMB_all"
derivSet = "v0.5"
gridName = "grid-default"

cosmoFisher = Config.get('fisher-'+fishSection,'saveSuffix')
from orphics.tools.io import listFromConfig
zs = listFromConfig(Config,gridName,'zrange')
z_edges = np.arange(zs[0],zs[1]+zs[2],zs[2])
zrange = (z_edges[1:]+z_edges[:-1])/2.
paramList = Config.get('fisher-'+fishSection,'paramList').split(',')
print paramList
paramLatexList = Config.get('fisher-'+fishSection,'paramLatexList').split(',')
fparams = {} 
for (key, val) in Config.items('params'):
    param = val.split(',')[0]
    fparams[key] = float(param)



"""RES STUDY"""
cmbfisher3 = getFisher("S4-3.0-0.4"+noatm,gridName,cal,cosmoFisher,paramList,derivSet)
cmbfisher5 = getFisher("S4-2.5-0.4"+noatm,gridName,cal,cosmoFisher,paramList,derivSet)
cmbfisher6 = getFisher("S4-2.0-0.4"+noatm,gridName,cal,cosmoFisher,paramList,derivSet)
cmbfisher7 = getFisher("S4-1.5-0.4"+noatm,gridName,cal,cosmoFisher,paramList,derivSet)
cmbfisher8 = getFisher("S4-1.0-0.4"+noatm,gridName,cal,cosmoFisher,paramList,derivSet)
fplots = FisherPlots(paramList,paramLatexList,fparams)
fplots.addFisher('cmb3',cmbfisher3)
fplots.addFisher('cmb5',cmbfisher5)
fplots.addFisher('cmb6',cmbfisher6)
fplots.addFisher('cmb7',cmbfisher7)
fplots.addFisher('cmb8',cmbfisher8)

#paramList = ['mnu','wa','w0','b_ym','tau','H0']
paramList = ['mnu','H0','tau','b_ym','alpha_ym','Ysig','gamma_ym','beta_ym','gammaYsig','betaYsig','wa','w0']
fplots.plotTri(paramList,['cmb3','cmb5','cmb6','cmb7','cmb8'],labels=['S4-3.0-0.4','S4-2.5-0.4','S4-2.0-0.4','S4-1.5-0.4','S4-1.0-0.4'],saveFile=out_dir+"Fig2DBeam.png",loc='upper right')
