import matplotlib
matplotlib.use('Agg')
from ConfigParser import SafeConfigParser 
import cPickle as pickle
import numpy as np
import sys
from orphics.tools.io import dictFromSection, listFromConfig
from orphics.tools.io import Plotter
import matplotlib.pyplot as plt
from szar.fisher import getFisher
from szar.counts import rebinN


expName = sys.argv[1]
gridName = sys.argv[2]
calName = sys.argv[3]
fishName = sys.argv[4]

iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)


fishSection = 'fisher-'+fishName
paramList = Config.get(fishSection,'paramList').split(',')
paramLatexList = Config.get(fishSection,'paramLatexList').split(',')
saveName = Config.get(fishSection,'saveSuffix')


FisherTot, paramList = sfisher.cluster_fisher_from_config(Config,expName,gridName,calName,fishName)


from collections import OrderedDict
priorList = OrderedDict()
priorList['tau'] = 0.1
priorList['H0'] = 10.0
priorList['omch2'] = 0.002
priorList['ombh2'] = 0.00023
priorList['ns'] = 0.006
priorList['As'] = 5.e-12
priorList['alpha_ym'] = 0.179
priorList['b_ym'] = 0.08
priorList['beta_ym'] = 0.1
priorList['gamma_ym'] = 0.1
priorList['Ysig'] = 0.0127
priorList['gammaYsig'] = 0.1
priorList['betaYsig'] = 1.0


import os
if 'mnu' in fishName:            
    pl = Plotter(labelY="$\sigma("+paramLatexList[paramList.index("mnu")]+")$",labelX="Iteration",ftsize=12)
elif 'w0' in fishName:
    pl = Plotter(labelY="$\\frac{\sigma("+paramLatexList[paramList.index("w0")]+")}{"+paramLatexList[paramList.index("w0")]+"}\%$",labelX="Iteration",ftsize=12)


for doBAO in [False,True]:    

    priorNameList = []
    priorValueList = []
    iterations = 0

    numlogs = 30
    pertol = 0.1
    mink = 5
    perRange = np.logspace(-4,2,numlogs)[::-1]



    for prior in priorList.keys():
        priorNameList.append(prior)

        preVal = np.inf
        priorRange = perRange*priorList[prior]/100.
        priorValueList.append(priorRange[0])
        print priorNameList, priorValueList
        sigs = []
        xs = []
        k = 0
        for val in priorRange:
            iterations += 1
            xs.append(iterations)
            FisherTot = 0.
            priorValueList[-1] = val

            ##########################
            # Populate Fisher
            Fisher = getFisher(N_fid,paramList,priorNameList,priorValueList,bigDataDir,saveId,pzcutoff,z_edges,fsky)
            ##########################


            
            fisherBAO = Fisher.copy()*0.
            if doBAO:
                if baoFile!='':
                    try:
                        fisherBAO = np.loadtxt(baoFile)
                    except:
                        fisherBAO = np.loadtxt(baoFile,delimiter=',') 
                    fisherBAO = np.pad(fisherBAO[:numCosmo,:numCosmo],pad_width=((0,numLeft),(0,numLeft)),mode="constant",constant_values=0.) # !!!!
                        

            FisherTot = Fisher + fisherPlanck
            FisherTot += fisherBAO


            Finv = np.linalg.inv(FisherTot)

            errs = np.sqrt(np.diagonal(Finv))
            errDict = {}
            for i,param in enumerate(paramList):
                errDict[param] = errs[i]

            if 'mnu' in fishName:            
                constraint = errDict['mnu']*1000
            elif 'w0' in fishName:
                constraint = errDict['w0']*100
            sigs.append(constraint)
            if (np.abs(preVal-constraint)*100./constraint)<pertol:
                print (constraint-preVal)*100./constraint
                if k>mink: break
            preVal = constraint
            print prior, val,constraint
            k+=1

        priorLabel = paramLatexList[paramList.index(prior)]
        if doBAO:
            lab = None
            lss = "--"
        else:
            lab = "$"+priorLabel+"$"
            lss = "-"

        pl.add(xs,sigs,label=lab,ls=lss)
    plt.gca().set_color_cycle(None)

pl.legendOn(loc='upper right',labsize=8)
pl.done(os.environ['WWW']+"paper/FigPrior_"+fishName+"_tau.pdf")
