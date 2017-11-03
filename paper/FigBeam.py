from orphics.tools.io import Plotter
import flipper.liteMap as lm
from szar.counts import ClusterCosmology
from orphics.tools.io import dictFromSection,listFromConfig
from configparser import SafeConfigParser 
from alhazen.halos import NFWMatchedFilterSN
import numpy as np
from orphics.tools.cmb import loadTheorySpectraFromCAMB
from alhazen.quadraticEstimator import NlGenerator,getMax
import sys, os

out_dir = os.environ["WWW"]+"paper/"
Mexp = np.log10(2.e14)
z = 0.7
c = 1.18

overdensity=500.
critical=True
atClusterZ=True


deg = 5.
px = 1.0

# deg = 1. # !!!!
# px = 2.0


dbeam = 0.1
beamList = np.arange(0.5,5.0+dbeam,dbeam)

expName = "S4-1.0-paper"
freq_to_use = 150.

# Mexp = np.log10(2.e14)
# z = 0.7
# c = 3.2

# overdensity=180.
# critical=False
# atClusterZ=False


clusterParams = 'cluster_params' # from ini file
cosmologyName = 'params' # from ini file

iniFile = "../szar/input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)


cosmoDict = dictFromSection(Config,cosmologyName)
constDict = dictFromSection(Config,'constants')
clusterDict = dictFromSection(Config,clusterParams)


bigDataDir = Config.get('general','bigDataDirectory')


beam = listFromConfig(Config,expName,'beams')
noise = listFromConfig(Config,expName,'noises')
freq = listFromConfig(Config,expName,'freqs')
lkneeT,lkneeP = listFromConfig(Config,expName,'lknee')
alphaT,alphaP = listFromConfig(Config,expName,'alpha')
tellmin,tellmax = listFromConfig(Config,expName,'halo_tellrange')
pellmin,pellmax = listFromConfig(Config,expName,'halo_pellrange')
try:
    doFg = Config.getboolean(expName,'do_foregrounds')
except:
    print("NO FG OPTION FOUND IN INI. ASSUMING TRUE.")
    doFg = True

ind = np.where(np.isclose(freq,freq_to_use))
beamFind = np.array(beam)[ind]
noiseFind = np.array(noise)[ind]
assert beamFind.size==1
assert noiseFind.size==1


testFreq = freq_to_use
from szar.foregrounds import fgNoises
fgs = fgNoises(constDict,ksz_battaglia_test_csv="data/ksz_template_battaglia.csv",tsz_battaglia_template_csv="data/sz_template_battaglia.csv")
tcmbmuk = constDict['TCMB'] * 1.0e6
ksz = lambda x: fgs.ksz_temp(x)/x/(x+1.)*2.*np.pi/ tcmbmuk**2.
radio = lambda x: fgs.rad_ps(x,testFreq,testFreq)/x/(x+1.)*2.*np.pi/ tcmbmuk**2.
cibp = lambda x: fgs.cib_p(x,testFreq,testFreq) /x/(x+1.)*2.*np.pi/ tcmbmuk**2.
cibc = lambda x: fgs.cib_c(x,testFreq,testFreq)/x/(x+1.)*2.*np.pi/ tcmbmuk**2.
tsz = lambda x: fgs.tSZ(x,testFreq,testFreq)/x/(x+1.)*2.*np.pi/ tcmbmuk**2.

fgFunc =  lambda x: ksz(x)+radio(x)+cibp(x)+cibc(x)+tsz(x)

beamTX = 5.0
noiseTX = 42.0
tellminX = 2
tellmaxX = 3000
lkneeTX = 0
alphaTX = 1
fgFuncX = None


noiseTY = noiseFind
noisePX = np.sqrt(2.)*noiseTY
noisePY = np.sqrt(2.)*noiseTY
pellminX = pellmin
pellmaxX = pellmax
pellminY = pellmin
pellmaxY = pellmax
tellminY = tellmin
tellmaxY = tellmax
lkneeTY = lkneeT
lkneePX = lkneePY = lkneeP
alphaTY = alphaT
alphaPX = alphaPY = alphaP



import flipper.liteMap as lm
from alhazen.quadraticEstimator import NlGenerator,getMax
dell = 10
gradCut = 2000
kellmin = 10
lmap = lm.makeEmptyCEATemplate(raSizeDeg=deg, decSizeDeg=deg,pixScaleXarcmin=px,pixScaleYarcmin=px)
kellmax = max(tellmax,pellmax)
from orphics.theory.cosmology import Cosmology
cc = Cosmology(lmax=int(kellmax),pickling=True)
theory = cc.theory
bin_edges = np.arange(kellmin,kellmax,dell)
myNls = NlGenerator(lmap,theory,bin_edges,gradCut=gradCut)

from scipy.interpolate import interp1d

from orphics.tools.io import Plotter
ellkk = np.arange(2,9000,1)
Clkk = theory.gCl("kk",ellkk)    
clfunc = interp1d(ellkk,Clkk,bounds_error=False,fill_value="extrapolate")

kellmax = 8000

cc = ClusterCosmology(cosmoDict,constDict,kellmax,pickling=True)
theory = cc.theory

pl = Plotter(labelX="Beam (arcmin)",labelY="$\\sigma(M)/M$ for $N=1000$",ftsize=16)

for miscenter in [False,True]:
    for lensName,linestyle in zip(["CMB_all","CMB_pol"],["-","--"]): 
        for doFg in [False,True]:
            
            if lensName=="CMB_pol" and not(doFg): continue
            if lensName=="CMB_all" and not(doFg) and miscenter: continue
            sns = []
            for beamNow in beamList:


                pols = Config.get(lensName,'polList').split(',')

                if doFg:
                    fgFuncY = fgFunc
                else:
                    fgFuncY = None

                beamPX = beamTY = beamPY = beamNow
                beamY = beamTY


                nTX,nPX,nTY,nPY = myNls.updateNoiseAdvanced(beamTX,noiseTX,beamPX,noisePX,tellminX,tellmaxX,pellminX,pellmaxX,beamTY,noiseTY,beamPY,noisePY,tellminY,tellmaxY,pellminY,pellmaxY,(lkneeTX,lkneePX),(alphaTX,alphaPX),(lkneeTY,lkneePY),(alphaTY,alphaPY),None,None,None,None,None,None,None,None,fgFuncX,fgFuncY,None,None,None,None,None,None,None,None)


                ls,Nls,ells,dclbb,efficiency = myNls.getNlIterative(pols,kellmin,kellmax,tellmax,pellmin,pellmax,dell=dell,halo=True)

                ls = ls[1:-1]
                Nls = Nls[1:-1]


                clkk_binned = clfunc(ls)



                Nls += clkk_binned



                if miscenter:
                    ray = beamNow/2.
                else:
                    ray = None


                sn,k,std = NFWMatchedFilterSN(cc,Mexp,c,z,ells=ls,Nls=Nls,kellmax=kellmax,overdensity=overdensity,critical=critical,atClusterZ=atClusterZ,rayleighSigmaArcmin=ray)

                print((sn*np.sqrt(1000.)))
                sns.append(1./(sn*np.sqrt(1000.)))



            fgpart = ""
            mispart = ""
            if miscenter:
                mispart = ", miscentered"
                col = "C0"
            else:
                col = "C1"
            if lensName=="CMB_all":
                lenspart = "T+P"
            else:
                lenspart = "P only"
            if not(doFg):
                fgpart = ", no foregrounds"
                col = "black"
                al = 0.5
            else:
                al=1.0
                
            lab = lenspart + fgpart + mispart
            pl.add(beamList,sns,label=lab,ls=linestyle,alpha=al,color=col)
pl.legendOn(loc="upper left",labsize=12)
pl.done(out_dir+"FigBeam.pdf")
