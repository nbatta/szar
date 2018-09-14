from __future__ import print_function
import szar.sims as s
import orphics.tools.io as io
from configparser import SafeConfigParser
# from enlib import enmap,utils,lensing,powspec
import os, sys
import numpy as np
import matplotlib.pyplot as plt

out_dir = os.environ['WWW']+"plots/"
iniFile = "input/params.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)
constDict = io.dictFromSection(Config,'constants')

sim = s.BattagliaSims(constDict)
nbins = 5
all = []
for snap in range(35,55):
    print((len(sim.trueM500overh[str(snap)] )))
    plt.clf()
    all.append(np.log10(sim.trueM500overh[str(snap)]).flatten())
    n, bins, patches = plt.hist(np.log10(sim.trueM500overh[str(snap)]) , nbins, alpha=0.5)
    
    plt.xlabel("log $M$")
    plt.ylabel("$N(M)$")
    plt.savefig(out_dir+"hist"+str(snap)+".png")


plt.clf()
n, bins, patches = plt.hist(np.asarray(all).flatten() , nbins, alpha=0.5)
    
plt.xlabel("log $M$")
plt.ylabel("$N(M)$")
plt.savefig(out_dir+"histall.png")
