import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ConfigParser import SafeConfigParser 
import cPickle as pickle
import sys
import numpy as np


def getFisher(bigDataDir,expName,gridName,calName,saveName,version):
    saveId = expName + "_" + gridName+ "_" + calName + "_v" + version

    paramList,FisherTot = pickle.load(open(bigDataDir+"savedS8Fisher_"+saveId+"_"+saveName+".pkl",'rb'))
    return FisherTot




iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)
bigDataDir = Config.get('general','bigDataDirectory')

fishSection = "lcdm"
noatm = ""
#noatm = "-noatm"
cal = "CMB_all"
#cal = "owl2"
#derivSet = "0.3_ysig_0.127"
derivSet = "0.5"
gridName = "grid-default"
#gridName = "grid-1.2"


cosmoFisher = Config.get('fisher-'+fishSection,'saveSuffix')
origParams = Config.get('fisher-'+fishSection,'paramList').split(',')


"""RES STUDY"""
#cmbfisher3 = getFisher(bigDataDir,"S4-3.0-0.4"+noatm,gridName,cal,cosmoFisher,derivSet)
#cmbfisher2 = getFisher(bigDataDir,"S4-2.5-0.4"+noatm,gridName,cal,cosmoFisher,derivSet)
#cmbfisher15 = getFisher(bigDataDir,"S4-2.0-0.4"+noatm,gridName,cal,cosmoFisher,derivSet)
#cmbfisher1 = getFisher(bigDataDir,"S4-1.5-0.4"+noatm,gridName,cal,cosmoFisher,derivSet)
#cmbfisher0 = getFisher(bigDataDir,"S4-1.0-0.4"+noatm,gridName,cal,cosmoFisher,derivSet)

cmbfisher1 = getFisher(bigDataDir,"S4-lowres"+noatm,gridName,cal,cosmoFisher,derivSet)
cmbfisher0 = getFisher(bigDataDir,"S4-highres"+noatm,gridName,cal,cosmoFisher,derivSet)

# cmbfisher3 = getFisher(bigDataDir,"S4-1.5-0.05"+noatm,gridName,cal,cosmoFisher,derivSet)
# cmbfisher2 = getFisher(bigDataDir,"S4-1.5-0.1"+noatm,gridName,cal,cosmoFisher,derivSet)
# cmbfisher15 = getFisher(bigDataDir,"S4-1.5-0.2"+noatm,gridName,cal,cosmoFisher,derivSet)
# cmbfisher1 = getFisher(bigDataDir,"S4-1.5-0.3"+noatm,gridName,cal,cosmoFisher,derivSet)
# cmbfisher0 = getFisher(bigDataDir,"S4-1.5-0.4"+noatm,gridName,cal,cosmoFisher,derivSet)
# cmbfisher01 = getFisher(bigDataDir,"S4-1.5-0.7"+noatm,gridName,cal,cosmoFisher,derivSet)

# cmbfisher3 = getFisher(bigDataDir,"SO-3m"+noatm,gridName,cal,cosmoFisher,derivSet)
# cmbfisher2 = getFisher(bigDataDir,"SO-5m"+noatm,gridName,cal,cosmoFisher,derivSet)
# cmbfisher15 = getFisher(bigDataDir,"SO-6m"+noatm,gridName,cal,cosmoFisher,derivSet)
# cmbfisher1 = getFisher(bigDataDir,"SO-7m"+noatm,gridName,cal,cosmoFisher,derivSet)


from szar.counts import getA
from orphics.tools.io import Plotter,dictFromSection,listFromConfig
fparams = {}   # the 
for (key, val) in Config.items('params'):
    if ',' in val:
        param, step = val.split(',')
        fparams[key] = float(param)
    else:
        fparams[key] = float(val)

constDict = dictFromSection(Config,'constants')
zs = listFromConfig(Config,gridName,'zrange')
z_edges = np.arange(zs[0],zs[1]+zs[2],zs[2])
zrange = (z_edges[1:]+z_edges[:-1])/2.

s81,As1 = getA(fparams,constDict,zrange)
print s81
s81zs = As1*s81
fparams['w0']=-0.97
s82,As2 = getA(fparams,constDict,zrange)
print s82
s82zs = As2*s82


outDir = "/gpfs01/astro/www/msyriac/paper/"


#zbins = np.append(np.arange(0.,1.5,0.5),3.0)
#zbins = np.append(np.arange(2.,3.0,0.2),3.0)
zbins = np.array([0.,0.5,1.0,1.5,3.0])#np.arange(1.5,3.0,0.1)
#zbins = zrange

#zbins = np.append(np.arange(2.,2.5,0.5),3.0)

#zbins = np.arange(0.,3.5,0.5)
#zbins = np.arange(0.,3.1,0.1)

pl = Plotter(labelX="$z$",labelY="$\sigma_8(z)/\sigma_8(z)_{w=-1}$",ftsize=12)
#pl = Plotter(labelX="$z$",labelY="$\sigma_8(z)$",ftsize=12)
#pl = Plotter(labelX="$z$",labelY="$D(z)/D(z)_{w=-1}$",ftsize=12)


colList = ['C1','C0','C2','C3','C4','C5']
#['coral','forestgreen','gold','indigo','purple']

from matplotlib.patches import Rectangle
currentAxis = plt.gca()
#for i,(f,lab,col) in enumerate(zip([cmbfisher3,cmbfisher15,cmbfisher0],['3.0\'','2.0\'','1.0\''],colList)):
#for i,(f,lab,col) in enumerate(zip([cmbfisher3,cmbfisher15,cmbfisher1,cmbfisher0],['3.0\'','2.0\'','1.5\'','1.0\''],colList)):
for i,(f,lab,col) in enumerate(zip([cmbfisher1,cmbfisher0],['1.5\' 0.85uK\'','1.0\' 0.95uK\''],colList)):
#for i,(f,lab,col) in enumerate(zip([cmbfisher3,cmbfisher2,cmbfisher15,cmbfisher1,cmbfisher0],['3.0\'','2.5\'','2.0\'','1.5\'','1.0\''],colList)):
#for i,(f,lab,col) in enumerate(zip([cmbfisher3,cmbfisher2,cmbfisher15,cmbfisher1,cmbfisher0,cmbfisher01],['0.05','0.1','0.2','0.3','0.4','0.7'],colList)):
#for i,(f,lab,col) in enumerate(zip([cmbfisher3,cmbfisher2,cmbfisher15,cmbfisher1],['SO-SZ + CMB Halo Lensing (P only) 3m','5m','6m','7m'],['coral','forestgreen','gold','indigo'])):
    inv = np.linalg.inv(f)
    err = np.sqrt(np.diagonal(inv))[len(origParams):]
    #print err
    #if lab=='0.1': continue #print err
    zcents = []
    errcents = []
    xerrs = []
    ms8 = []
    pad = 0.05
    s80mean = np.mean(s81zs[np.logical_and(zrange>=zbins[0],zrange<zbins[1])])
    yerrsq0 = (1./sum([1/x**2. for x in err[np.logical_and(zrange>=zbins[0],zrange<zbins[1])]]))

    for zleft,zright in zip(zbins[:-1],zbins[1:]):
        errselect = err[np.logical_and(zrange>=zleft,zrange<zright)]
        zcent = (zleft+zright)/2.
        zcents.append(zcent)
        yerr = np.sqrt(1./sum([1/x**2. for x in errselect]))
        xerr = (zright-zleft)/2.
        xerrs.append(xerr)
        s8now = np.mean(s81zs[np.logical_and(zrange>=zleft,zrange<zright)])
        print lab,zleft,zright, yerr,s8now, yerr*100./s8now, "%"
        #s8now = np.mean(s81zs[np.logical_and(zrange>=zleft,zrange<zright)])/s81
        #yerrsq = (1./sum([1/x**2. for x in errselect]))
        #yerr = (s8now/s80mean)*np.sqrt(yerrsq/s8now**2. + yerrsq0/s80mean**2.)
        errcents.append(yerr)
        ms8.append(s8now)
        currentAxis.add_patch(Rectangle((zcent - xerr+pad, 1. - yerr/s8now), 2*xerr-pad/2., 2.*yerr/s8now, facecolor=col))
    print "====================="
    pl._ax.fill_between(zrange, 1., 1.,label=lab,alpha=0.75,color=col)
    

#pl.add(zrange,s82zs/s81zs,label="$w=-0.97$",color='red',alpha=0.5)
pl.add(zrange,s81zs/s81zs,color='black',alpha=0.5,ls="--")#,label="$w=-1$")

# pl.add(zrange,s82zs/s81zs/s82*s81,label="$w=-0.97$",color='red',alpha=0.5)
# pl.add(zrange,s81zs*0.+1.,label="$w=-1$",color='black',alpha=0.5,ls="--")


pl.legendOn(labsize=9)
#pl._ax.set_ylim(0.9,1.1) # res
#pl._ax.set_ylim(0.95,1.05) # fsky
#pl._ax.text(0.8,.82,"Madhavacheril et. al. in prep")
pl.done(outDir+"cdts8.png")
#pl.done(outDir+"s8SO.png")
