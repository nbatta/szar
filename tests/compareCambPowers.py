"""
This script examines the outputs of CAMB_wa/out.ini.
It looks at the derivatives of wa to make sure
they are well behaved.

Feb 24, 2017 - MM 
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
from orphics.tools.io import Plotter



zrange = np.arange(0.,3.05,0.05)

pl = Plotter(scaleY='log',scaleX='log')

for stepSize,ls in zip([0.2,0.1,0.05],['-','--','-.']):
    cambOutUpFile = lambda z: "/home/msyriac/software/CAMB_wa/forDerivsStep"+str(stepSize)+"Up_matterpower_"+str(z)+".dat"
    cambOutDnFile = lambda z: "/home/msyriac/software/CAMB_wa/forDerivsStep"+str(stepSize)+"Dn_matterpower_"+str(z)+".dat"

    for i,z in enumerate(zrange[::-1]):

        kh_camb_up,P_camb_up = np.loadtxt(cambOutUpFile(z),unpack=True)
        kh_camb_dn,P_camb_dn = np.loadtxt(cambOutDnFile(z),unpack=True)

        assert np.all(kh_camb_dn==kh_camb_up)

        Pderiv = np.abs(P_camb_up-P_camb_dn)/stepSize

        if i%5==0:
            pl.add(kh_camb_up,Pderiv,ls=ls)
            #pl.add(kh_camb,P_camb)
            #pl.add(kh,pk[Nzs-i-1,:],ls="--")

pl.done("output/testwa.png")
