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
import szar.fisher as sfisher



expName = sys.argv[1]
gridName = sys.argv[2]
calName = sys.argv[3]
fishName = sys.argv[4]

iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)

bigDataDir = Config.get('general','bigDataDirectory')

fishSection = 'fisher-'+fishName
paramList = Config.get(fishSection,'paramList').split(',')
paramLatexList = Config.get(fishSection,'paramLatexList').split(',')
saveName = Config.get(fishSection,'saveSuffix')
version = Config.get('general','version') 
pzcutoff = Config.getfloat('general','photoZCutOff')
fsky = Config.getfloat(expName,'fsky')

zs = listFromConfig(Config,gridName,'zrange')
z_edges = np.arange(zs[0],zs[1]+zs[2],zs[2])

saveId = sfisher.save_id(expName,gridName,calName,version)
derivRoot = sfisher.deriv_root(bigDataDir,saveId)
# Fiducial number counts
new_z_edges, N_fid = sfisher.rebinN(np.load(sfisher.fid_file(bigDataDir,saveId)),pzcutoff,z_edges)
N_fid = N_fid*fsky
print "Effective number of clusters: ", N_fid.sum()



from collections import OrderedDict
priorList = OrderedDict()
priorList['tau'] = 0.01
priorList['H0'] = 10.0
priorList['omch2'] = 0.002
priorList['ombh2'] = 0.00023
priorList['ns'] = 0.006
priorList['As'] = 5.e-12
priorList['alpha_ym'] = 0.179
priorList['b_ym'] = 0.08
priorList['beta_ym'] = 0.1
priorList['gamma_ym'] = 0.1
priorList['Ysig'] = 0.127
priorList['gammaYsig'] = 1.0
priorList['betaYsig'] = 1.0


import os
if 'mnu' in fishName:            
    pl = Plotter(labelY="$\sigma("+paramLatexList[paramList.index("mnu")]+")$ (meV)",labelX="Iteration",ftsize=20)
elif 'w0' in fishName:
    pl = Plotter(labelY="$\\frac{\sigma("+paramLatexList[paramList.index("w0")]+")}{"+paramLatexList[paramList.index("w0")]+"}\%$",labelX="Iteration",ftsize=20)


#for doBAO in [False,True]:    
for fishSection in ["fisher-"+fishName,"fisher-"+fishName+"-DESI"]:


    priorNameList = []
    priorValueList = []
    iterations = 0

    numlogs = 30
    pertol = 0.1
    mink = 5
    perRange = np.logspace(-4,2,numlogs)[::-1]
    #perRange = np.logspace(-8,2,numlogs)[::-1]



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
            Fisher = sfisher.getFisher(N_fid,paramList,priorNameList,priorValueList,derivRoot,pzcutoff,z_edges,fsky)
            ##########################

            # Number of non-SZ params (params that will be in Planck/BAO)
            numCosmo = Config.getint(fishSection,'numCosmo')
            numLeft = len(paramList) - numCosmo

            try:
                otherFishers = Config.get(fishSection,'otherFishers').split(',')
            except:
                traceback.print_exc()
                print "No other fishers found."
                otherFishers = []
            for otherFisherFile in otherFishers:
                try:
                    other_fisher = np.loadtxt(otherFisherFile)
                except:
                    other_fisher = np.loadtxt(otherFisherFile,delimiter=',')
                other_fisher = sfisher.pad_fisher(other_fisher,numLeft)
                Fisher += other_fisher

            


            try:
                do_cmb_fisher = Config.getboolean(fishSection,"do_cmb_fisher")
            except:
                do_cmb_fisher = False

            try:
                do_clkk_fisher = Config.getboolean(fishSection,"do_clkk_fisher")
            except:
                do_clkk_fisher = False


            if do_clkk_fisher:
                assert do_cmb_fisher, "Sorry, currently Clkk fisher requires CMB fisher to be True as well."
                lensName = Config.get(fishSection,"clkk_section")
            else:
                lensName = None

            if do_cmb_fisher:


                import pyfisher.clFisher as pyfish
                # Load fiducials and derivatives
                cmbDerivRoot = Config.get("general","cmbDerivRoot")
                cmbParamList = paramList[:numCosmo]


                cmb_fisher_loaded = False
                if True:
                    import time
                    hashval = sfisher.hash_func(cmbParamList,expName,lensName,do_clkk_fisher,time.strftime('%Y%m%d'))
                    pkl_file = "output/pickledFisher_"+hashval+".pkl"

                    try:
                        cmb_fisher = pickle.load(open(pkl_file,'rb'))
                        cmb_fisher_loaded = True
                        print "Loaded pickled CMB fisher."
                    except:
                        pass

                if not(cmb_fisher_loaded):
                    fidCls = np.loadtxt(cmbDerivRoot+'_fCls.csv',delimiter=',')
                    dCls = {}
                    for paramName in cmbParamList:
                        dCls[paramName] = np.loadtxt(cmbDerivRoot+'_dCls_'+paramName+'.csv',delimiter=',')

                    print "Calculating CMB fisher matrix..."
                    cmb_fisher = pyfish.fisher_from_config(fidCls,dCls,cmbParamList,Config,expName,lensName)
                    if True:
                        print "Pickling CMB fisher..."
                        pickle.dump(cmb_fisher,open(pkl_file,'wb'))


                cmb_fisher = sfisher.pad_fisher(cmb_fisher,numLeft)
            else:
                cmb_fisher = 0.    

            FisherTot = Fisher+cmb_fisher

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
        if "DESI" in fishSection:
            lab = None
            lss = "--"
        else:
            lab = "$"+priorLabel+"$"
            lss = "-"

        pl.add(xs,sigs,label=lab,ls=lss)
    plt.gca().set_color_cycle(None)

pl.legendOn(loc='upper right',labsize=11)
pl.done(os.environ['WWW']+"paper/FigPrior_"+fishName+"_tau.pdf")
