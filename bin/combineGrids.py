from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys,os
import pickle as pickle
import numpy as np
from configparser import SafeConfigParser 
import szar.fisher as sfisher

iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)

bigDataDir = Config.get('general','bigDataDirectory')
version = Config.get('general','version')

expName = sys.argv[1] #"S4-1.0-paper"
cmbGrid = "grid-default"
cmbCal = "CMB_all"
owlCal = "owl2"
saveCalName = "CMB_all_joint"

fidcmb_file = sfisher.mass_grid_name_cmb(bigDataDir,expName,cmbGrid,cmbCal,version)
upcmb_file = sfisher.mass_grid_name_cmb_up(bigDataDir,expName,cmbGrid,cmbCal,version)
dncmb_file = sfisher.mass_grid_name_cmb_dn(bigDataDir,expName,cmbGrid,cmbCal,version)


owl_file = sfisher.mass_grid_name_owl(bigDataDir,owlCal)




outDir = os.environ['WWW']+"paper/"



mindicesList = [60,80,120,160]

from orphics.io import Plotter
    
for cmbfile,cmbtype,save_func in zip([fidcmb_file,upcmb_file,dncmb_file],['fid','up','dn'], \
                                    [sfisher.mass_grid_name_cmb,sfisher.mass_grid_name_cmb_up,sfisher.mass_grid_name_cmb_dn]):


    grids = {}
    mmins = []
    mmaxes = []
    zmins = []
    zmaxes = []
    dms = []
    dzs = []

    pl = Plotter(labelX="$z$",labelY="S/N per cluster",ftsize=14)

    gridList = [cmbfile,owl_file]

    for gridFile,ls,lab,outPlot in zip(gridList,['-','--'],['CMB lensing','optical lensing'],['cmb','owl']):


        medges,zedges,errgrid = pickle.load(open(gridFile,'rb'))


        print(errgrid.shape)
        M_edges = 10**medges
        M = (M_edges[1:]+M_edges[:-1])/2.
        mexpgrid = np.log10(M)
        zgrid = (zedges[1:]+zedges[:-1])/2.

        if "cmb" in outPlot:
            outmedges = medges.copy()
            outzedges = zedges.copy()
            outmgrid = mexpgrid.copy()
            outzgrid = zgrid.copy()


        mmin = mexpgrid[0]
        mmax = mexpgrid[-1]
        zmin = zgrid[0]
        zmax = zgrid[-1]

        print(mmin,mmax,zmin,zmax)

        grids[gridFile] = (mexpgrid,zgrid,errgrid)

        mmins.append(mmin)
        mmaxes.append(mmax)
        zmins.append(zmin)
        zmaxes.append(zmax)
        dms.append(min(np.diff(mexpgrid)))
        dzs.append(min(np.diff(zgrid)))

        sngrid = 1./errgrid
        print(sngrid.shape)
        for ind in mindicesList:
            if "CMB" in lab:
                labadd = '{:02.1f}'.format(10**(mexpgrid[ind])/1e14)+" $10^{14}  M_{\odot}/h$"
            else:
                labadd = None
            pl.add(zgrid,sngrid[ind,:].ravel(),ls=ls,label=labadd)
            print(mexpgrid[ind])

        rtol = 1.e-3
        plt.gca().set_color_cycle(None)

    print(outmgrid[0],outmgrid[-1],outmgrid.shape)
    print(outzgrid[0],outzgrid[-1],outzgrid.shape)
    from orphics.maps import interpolateGrid

    jointgridsqinv = 0.
    for key in grids:

        inmgrid,inzgrid,inerrgrid = grids[key]
        outerrgrid = interpolateGrid(inerrgrid,inmgrid,inzgrid,outmgrid,outzgrid,regular=False,kind="cubic",bounds_error=False,fill_value=np.inf)

        mmin = outmgrid[0]
        mmax = outmgrid[-1]
        zmin = outzgrid[0]
        zmax = outzgrid[-1]



        jointgridsqinv += (1./outerrgrid**2.)


    jointgrid = np.sqrt(1./jointgridsqinv)
    snjoint = 1./jointgrid

    for ind in mindicesList:
        pl.add(outzgrid,snjoint[ind,:].ravel(),ls="-.")
        print(mexpgrid[ind])

    pl.legendOn(loc='upper right',labsize=10)
    pl.done(outDir+"slice"+cmbtype+".pdf")


    #pl.legendOn(loc='upper right',labsize=8)
    #pl.done(outDir+"slice.pdf")


    #from orphics.io import Plotter
    pgrid = np.rot90(1./jointgrid)
    pl = Plotter(labelX="$\\mathrm{log}_{10}(M)$",labelY="$z$",ftsize=14)
    pl.plot2d(pgrid,extent=[mmin,mmax,zmin,zmax],levels=[1.0,3.0,5.0],labsize=14)
    pl.done(outDir+"joint"+cmbtype+".png")


    #savefile = "/astro/astronfs01/workarea/msyriac/data/SZruns/v0.6/lensgrid_S4-1.0-0.4_grid-default_CMB_all_joint_v0.5.pkl"
    #savefile = "/astro/astronfs01/workarea/msyriac/data/SZruns/v0.6/lensgridRayUp_S4-1.0-0.4_grid-default_CMB_all_joint_v0.5.pkl"
    #savefile = "/astro/astronfs01/workarea/msyriac/data/SZruns/v0.6/lensgridRayDn_S4-1.0-0._grid-default_CMB_all_joint_v0.5.pkl"
    #pickle.dump((outmedges,outzedges,jointgrid),open(savefile,'wb'))

    
    save_file = save_func(bigDataDir,expName,cmbGrid,saveCalName,version)
    pickle.dump((outmedges,outzedges,jointgrid),open(save_file,'wb'))
