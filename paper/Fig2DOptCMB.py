from ConfigParser import SafeConfigParser 
import cPickle as pickle
import sys, os
import itertools

from orphics.tools.io import FisherPlots


def getFisher(expName,gridName,calName,saveName,inParamList,suffix):
    saveId = expName + "_" + gridName+ "_" + calName + "_" + suffix

    paramList,FisherTot = pickle.load(open(bigDataDir+"savedFisher_"+saveId+"_"+saveName+".pkl",'rb'))
    print paramList
    print inParamList
    #assert paramList==inParamList
    return FisherTot

out_dir = os.environ['WWW']+"paper/"


iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)
bigDataDir = Config.get('general','bigDataDirectory')


#fishSection = "mnu-w0-wa"
#fishSection = "mnu-w0"
#fishSection = "lcdm"


noatm = ""
cals = ["CMB_all","CMB_pol","owl2","owl1"]
labs = ["CMB T+P","CMB P only","Optical $z<2$","Optical $z<1$"]
cols = ["C0","C0","C1","C1"]
lss = ["-","--","-","--"]
derivSet = "v0.5"
gridNames = ["grid-default","grid-default","grid-owl2","grid-owl1"]

fplots = FisherPlots()
fplots.startFig() 

#for fishSection,alphas in zip(["mnu-w0-wa","mnu-w0"],[[1,1,1,1],[0.3,0.3,0.3,0.3]]):
#for fishSection,alphas in zip(["mnu-w0-wa"],[[1,1,1,1]]):
for fishSection,alphas in zip(["mnu-w0"],[[1,1,1,1]]):

    #if fishSection == "mnu-w0": labs = itertools.repeat(None)
    
    cosmoFisher = Config.get('fisher-'+fishSection,'saveSuffix')
    paramList = Config.get('fisher-'+fishSection,'paramList').split(',')
    paramLatexList = Config.get('fisher-'+fishSection,'paramLatexList').split(',')
    fparams = {} 
    for (key, val) in Config.items('params'):
        param = val.split(',')[0]
        fparams[key] = float(param)

    fplots.addSection(fishSection,paramList,paramLatexList,fparams)


    """RES STUDY"""
    for cal,gridName in zip(cals,gridNames):
        cmbfisher = getFisher("S4-1.0-0.4"+noatm,gridName,cal,cosmoFisher,paramList,derivSet)
        fplots.addFisher(fishSection,cal,cmbfisher.copy())



    #fplots.plotPair(fishSection,['mnu','w0'],cals,labels=labs,xlims=[-0.1,0.2],ylims=[-1.12,-0.88],cols=cols,lss=lss,loc='lower left',alphas=alphas)
    fplots.plotPair(fishSection,['mnu','w0'],cals,labels=labs,xlims=[-0.05,0.15],ylims=[-1.1,-0.86],cols=cols,lss=lss,loc='lower left',alphas=alphas)

fplots.done(saveFile=out_dir+"Fig2DOptCMBTalk.png")
