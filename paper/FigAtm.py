import numpy as np
from configparser import SafeConfigParser 
import szar.fisher as sfisher
from orphics.tools.io import dictFromSection, listFromConfig
import orphics.tools.io as io
import os,sys


out_dir = os.environ['WWW']+"paper/"

iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)
version = Config.get('general','version')
bigDataDir = Config.get('general','bigDataDirectory')

#expName = "S4-1.0-paper"
expList = ['S4-1.0-paper','S4-1.5-paper','S4-2.0-paper','S4-2.5-paper','S4-3.0-paper']
gridName = "grid-default"


# get mass and z grids
ms = listFromConfig(Config,gridName,'mexprange')
mexp_edges = np.arange(ms[0],ms[1]+ms[2],ms[2])
zs = listFromConfig(Config,gridName,'zrange')
z_edges = np.arange(zs[0],zs[1]+zs[2],zs[2])



lkneeList = np.arange(0,6000,500.)
alphaList = [-4.,-4.5,-5.]

pl = io.Plotter(scaleY='log',labelX="$\ell_{\mathrm{knee}}$",labelY="Cluster detections",ftsize=16)


lslist = ["--","-","-."]
collist = ['#1C110A','#E4D6A7','#E9B44C','#9B2915','#50A2A7']
lablist = ['S4 1.0\'','S4 1.5\'','S4 2.0\'','S4 2.5\'','S4 3.0\'']

for expName,col,lab in zip(expList,collist,lablist):
    for alpha,ls in zip(alphaList,lslist):
        Ns = []
        for lknee in lkneeList:

            try:
                N = sfisher.counts_from_config(Config,bigDataDir,version,expName,gridName,mexp_edges,z_edges,lkneeTOverride = lknee,alphaTOverride=alpha)
                print(N)
            except:
                N = np.nan
            Ns.append(N)

        labnow = lab if ls=="-" else None
        pl.add(lkneeList,Ns,label=labnow,color=col,ls=ls)


        print((alpha,(Ns[-1]-Ns[0])*100./Ns[0], " %"))

pl.legendOn(labsize=12,loc="lower left")
pl._ax.axvline(x=3500.,color="k",alpha=0.5,ls="--")
pl.done(out_dir+"FigAtm.pdf")

