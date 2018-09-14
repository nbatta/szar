from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
from configparser import SafeConfigParser 
import pickle as pickle
import sys, os
import numpy as np

from orphics.tools.io import FisherPlots


def getFisher(expName,gridName,calName,saveName,inParamList,suffix):
    saveId = expName + "_" + gridName+ "_" + calName + "_" + suffix

    paramList,FisherTot = pickle.load(open(bigDataDir+"savedFisher_"+saveId+"_"+saveName+".pkl",'rb'))
    try:
        assert paramList==inParamList
    except:
        print("ERROR")
        print(paramList)
        print(inParamList)
        sys.exit()
    return FisherTot


out_dir = os.environ['WWW']+"paper/"

iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)
bigDataDir = Config.get('general','bigDataDirectory')


fishSection = "mnu-w0-wa-paper"
#fishSection = "lcdm"

noatm = ""
cal = "CMB_all"
derivSet = "v0.6"
gridName = "grid-default"

cosmoFisher = Config.get('fisher-'+fishSection,'saveSuffix')
from orphics.tools.io import listFromConfig
zs = listFromConfig(Config,gridName,'zrange')
z_edges = np.arange(zs[0],zs[1]+zs[2],zs[2])
zrange = (z_edges[1:]+z_edges[:-1])/2.
paramList = Config.get('fisher-'+fishSection,'paramList').split(',')


if "CMB" in cal:
    paramList.append("sigR")

# if "owl" in cal:
#     if not("b_wl") in paramList:
#         paramList.append("b_wl")





print(paramList)
paramLatexList = Config.get('fisher-'+fishSection,'paramLatexList').split(',')
fparams = {} 
for (key, val) in Config.items('params'):
    param = val.split(',')[0]
    fparams[key] = float(param)



"""RES STUDY"""
cmbfisher3 = getFisher("S4-3.0-paper"+noatm,gridName,cal,cosmoFisher,paramList,derivSet)
cmbfisher5 = getFisher("S4-2.5-paper"+noatm,gridName,cal,cosmoFisher,paramList,derivSet)
cmbfisher6 = getFisher("S4-2.0-paper"+noatm,gridName,cal,cosmoFisher,paramList,derivSet)
cmbfisher7 = getFisher("S4-1.5-paper"+noatm,gridName,cal,cosmoFisher,paramList,derivSet)
cmbfisher8 = getFisher("S4-1.0-paper"+noatm,gridName,cal,cosmoFisher,paramList,derivSet)
#fplots = FisherPlots(paramList,paramLatexList,fparams)

fplots = FisherPlots()
fplots.startFig() 
fplots.addSection(fishSection,paramList,paramLatexList,fparams)

#fplots.addFisher('cmb3',cmbfisher3)
#fplots.addFisher('cmb5',cmbfisher5)
#fplots.addFisher('cmb6',cmbfisher6)
#fplots.addFisher('cmb7',cmbfisher7)
#fplots.addFisher('cmb8',cmbfisher8)
fplots.addFisher(fishSection,"cmb3",cmbfisher3.copy())
fplots.addFisher(fishSection,"cmb5",cmbfisher5.copy())
fplots.addFisher(fishSection,"cmb6",cmbfisher6.copy())
fplots.addFisher(fishSection,"cmb7",cmbfisher7.copy())
fplots.addFisher(fishSection,"cmb8",cmbfisher8.copy())

CB_color_cycle = ['#1C110A','#E4D6A7','#E9B44C','#9B2915','#50A2A7'][::-1]
import matplotlib as mpl
mpl.rcParams['axes.color_cycle'] = CB_color_cycle


#paramList = ['mnu','wa','w0','b_ym','tau','H0']
#paramList = ['mnu','H0','tau','b_ym','alpha_ym','Ysig','gamma_ym','beta_ym','gammaYsig','betaYsig','wa','w0']
paramList = ['mnu','H0','tau','b_ym','alpha_ym','Ysig','gamma_ym','beta_ym','gammaYsig','betaYsig','wa','w0']#,'b_wl']
#fplots.plotTri(fishSection,paramList,['cmb8'],labels=['S4-1.0-paper'],saveFile=out_dir+"Fig2DBeam.png",loc='upper right')

labList = ['S4 1.0\'','S4 1.5\'','S4 2.0\'','S4 2.5\'','S4 3.0\''][::-1]

fplots.plotTri(fishSection,paramList,['cmb3','cmb5','cmb6','cmb7','cmb8'],labels=labList,saveFile=out_dir+"Fig2DBeamTest.png",loc='upper right')
#fplots.plotTri(paramList,['cmb3','cmb5','cmb6','cmb7','cmb8'],labels=['S4-3.0-paper','S4-2.5-paper','S4-2.0-paper','S4-1.5-paper','S4-1.0-paper'],saveFile=out_dir+"Fig2DBeam.png",loc='upper right')


# def plotTri(self,section,paramList,setNames,cols=itertools.repeat(None),lss=itertools.repeat(None),labels=itertools.repeat(None),saveFile="default.png",levels=[2.],xlims=None,ylims=None,loc='upper right',centerMarker=True,**kwargs):

