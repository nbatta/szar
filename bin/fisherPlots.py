import matplotlib
matplotlib.use('Agg')
from configparser import SafeConfigParser 
import pickle as pickle
import sys

from orphics.tools.io import FisherPlots


def getFisher(expName,gridName,calName,saveName,inParamList,suffix):
    saveId = expName + "_" + gridName+ "_" + calName + "_" + suffix

    paramList,FisherTot = pickle.load(open(bigDataDir+"savedFisher_"+saveId+"_"+saveName+".pkl",'rb'))
    print(paramList)
    print(inParamList)
    #assert paramList==inParamList
    return FisherTot




iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)
bigDataDir = Config.get('general','bigDataDirectory')


#fishSection = "mnu-w0-wa"
fishSection = "lcdm"

noatm = ""
#noatm = "-noatm"
#cal = "CMB_all_miscentered"
cal = "CMB_all"
#cal = "owl2"
#derivSet = "v0.3_ysig_0.127"
derivSet = "v0.5"
gridName = "grid-default"

cosmoFisher = Config.get('fisher-'+fishSection,'saveSuffix')
paramList = Config.get('fisher-'+fishSection,'paramList').split(',')
paramLatexList = Config.get('fisher-'+fishSection,'paramLatexList').split(',')


if "CMB" in cal:
    if not("sigR") in paramList:
        paramList.append("sigR")
        paramLatexList.append("sigma_R")
elif "owl" in cal:
    if not("b_wl") in paramList:
        paramList.append("b_wl")
        paramLatexList.append("b_{w}")
   


fparams = {} 
for (key, val) in Config.items('params'):
    param = val.split(',')[0]
    fparams[key] = float(param)



"""RES STUDY"""
#cmbfisher3 = getFisher("S4-3.0-0.4"+noatm,gridName,cal,cosmoFisher,paramList,derivSet)
cmbfisher3 = getFisher("S4-1.0-0.4"+noatm,gridName,cal,cosmoFisher,paramList,derivSet)
# cmbfisher5 = getFisher("S4-2.5-0.4"+noatm,gridName,cal,cosmoFisher,paramList,derivSet)
#cmbfisher6 = getFisher("S4-2.0-0.4"+noatm,gridName,cal,cosmoFisher,paramList,derivSet)
# cmbfisher7 = getFisher("S4-1.5-0.4"+noatm,gridName,cal,cosmoFisher,paramList,derivSet)
#cmbfisher8 = getFisher("S4-1.0-0.4"+noatm,gridName,cal,cosmoFisher,paramList,derivSet)

fplots = FisherPlots()
fplots.startFig() 

fplots.addSection(fishSection,paramList,paramLatexList,fparams)
fplots.addFisher(fishSection,'cmb3',cmbfisher3.copy())


# fplots.addFisher('cmb5',cmbfisher5)
#fplots.addFisher('cmb6',cmbfisher6)
# fplots.addFisher('cmb7',cmbfisher7)
#fplots.addFisher('cmb8',cmbfisher8)

#fplots.plotTri(['mnu','wa','w0','b_ym','tau','H0'],['cmb3','cmb6','cmb8'],labels=['S4-3.0-0.4','S4-2.0-0.4','S4-1.0-0.4'],saveFile="/gpfs01/astro/www/msyriac/test.png",loc='upper right')
#fplots.plotTri(['mnu','wa','w0','b_ym'],['cmb3','cmb6','cmb8'],labels=['S4-3.0-0.4','S4-2.0-0.4','S4-1.0-0.4'],saveFile="/gpfs01/astro/www/msyriac/test.png",loc='upper right')
#fplots.plotTri(['omch2','S8All','H0'],['cmb3'],labels=['SO-v2'],saveFile="/gpfs01/astro/www/msyriac/test.png",loc='upper right')

fplots.plotTri(fishSection,paramList,['cmb3'],labels=['SO-v2'],saveFile="/gpfs01/astro/www/msyriac/test.png",loc='upper right')


#fplots.plotPair(['mnu','w0'],['cmb3','cmb5','cmb6','cmb7','cmb8'],labels=['S4-3.0-0.4','S4-2.5-0.4','S4-2.0-0.4','S4-1.5-0.4','S4-1.0-0.4'],saveFile="/gpfs01/astro/www/msyriac/s4resatmwa_"+cal+noatm+"_"+cosmoFisher+"_"+derivSet+".png",xlims=[-0.1,0.2],ylims=[-1.12,-0.88])
#fplots.plotPair(['mnu','w0'],['cmb3','cmb5','cmb6','cmb7','cmb8'],labels=['S4-3.0-0.4','S4-2.5-0.4','S4-2.0-0.4','S4-1.5-0.4','S4-1.0-0.4'],saveFile="/gpfs01/astro/www/msyriac/s4resatmmnuwa_"+cal+noatm+"_"+cosmoFisher+"_"+derivSet+".png",loc='lower left')#,xlims=[-0.1,0.2])#,ylims=[-1.12,-0.88])
#fplots.plotPair(['w0','wa'],['cmb7'],labels=['S4-1.5-0.4'],saveFile="/gpfs01/astro/www/msyriac/s4resatmwa_"+cal+noatm+"_"+cosmoFisher+"_"+derivSet+".png")#,xlims=[-0.1,0.2],ylims=[-1.12,-0.88])

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

# res = "5m"
# owl1x = getFisher("S4-5m"+noatm,'owl2',cosmoFisher,paramList,"wstep")
# owl2x = getFisher("S4-5m"+noatm,'owl2',cosmoFisher,paramList,"wstep2x")
# fplots = FisherPlots(paramList,paramLatexList,fparams)
# fplots.addFisher('owl1x',owl1x)
# fplots.addFisher('owl2x',owl2x)
# listFishers = ['owl1x','owl2x']
# labFishers = ['S4-5m LSST-like','S4-5m LSST-like 2x error']

# fplots.plotPair(['mnu','w0'],listFishers,labels=labFishers,saveFile="/gpfs01/astro/www/msyriac/s42d_mnuw0_2x"+res+"_"+noatm+"_"+cosmoFisher+"_"+derivSet+".png")


#"""CMB OWL comparison"""

# res = "7m"
# owl1 = getFisher("S4-"+res+noatm,'grid-owl2','owl2',cosmoFisher,paramList,derivSet)
# fplots = FisherPlots(paramList,paramLatexList,fparams)
# fplots.addFisher('owl1',owl1)
# listFishers = ['owl1']
# labFishers = ['S4-'+res]

# fplots.plotPair(['b_wl','b_ym'],listFishers,labels=labFishers,saveFile="/gpfs01/astro/www/msyriac/s42d_bb_"+res+"_"+noatm+"_"+cosmoFisher+"_"+derivSet+".png")
