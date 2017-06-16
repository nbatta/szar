import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys,os
import cPickle as pickle
import numpy as np

gridList = sys.argv[1:]
outName = sys.argv[-1]

outDir = os.environ['WWW']

print gridList
print outName

assert outName[-3:]!=".py"

grids = {}
mmins = []
mmaxes = []
zmins = []
zmaxes = []
dms = []
dzs = []

mindicesList = [60,80,120,160]

from orphics.tools.io import Plotter
pl = Plotter(labelX="$z$",labelY="S/N per cluster",ftsize=14)

for gridFile,ls,lab,outPlot in zip(gridList,['-','--'],['CMB lensing','optical lensing'],['cmb','owl']):

    filen = gridFile #"data/"+gridFile+".pkl"
    
    medges,zedges,errgrid = pickle.load(open(filen,'rb'))

    
    print errgrid.shape
    M_edges = 10**medges
    M = (M_edges[1:]+M_edges[:-1])/2.
    mexpgrid = np.log10(M)
    zgrid = (zedges[1:]+zedges[:-1])/2.

    if "cmb" in outPlot:
        outmedges = medges.copy()
        outzedges = zedges.copy()
        outmgrid = mexpgrid.copy()
        outzgrid = zgrid.copy()
    # print outmgrid[0],outmgrid[-1],outmgrid.shape
    # print outzgrid[0],outzgrid[-1],outzgrid.shape


    mmin = mexpgrid[0]
    mmax = mexpgrid[-1]
    zmin = zgrid[0]
    zmax = zgrid[-1]

    print mmin,mmax,zmin,zmax

    grids[gridFile] = (mexpgrid,zgrid,errgrid)

    mmins.append(mmin)
    mmaxes.append(mmax)
    zmins.append(zmin)
    zmaxes.append(zmax)
    dms.append(min(np.diff(mexpgrid)))
    dzs.append(min(np.diff(zgrid)))

    sngrid = 1./errgrid
    print sngrid.shape
    for ind in mindicesList:
        if "CMB" in lab:
            labadd = '{:02.1f}'.format(10**(mexpgrid[ind])/1e14)+" $10^{14}  M_{\odot}/h$"
        else:
            labadd = None
        pl.add(zgrid,sngrid[ind,:].ravel(),ls=ls,label=labadd)
        print mexpgrid[ind]
    
    # pgrid = np.rot90(sngrid)
    # pl = Plotter(labelX="$\\mathrm{log}_{10}(M)$",labelY="$z$",ftsize=14)
    # pl.plot2d(pgrid,extent=[mmin,mmax,zmin,zmax],levels=[3.0,5.0],labsize=14,aspect="auto")
    # pl.done(outDir+outPlot+".png")

    rtol = 1.e-3
    # pl.add(zgrid,sngrid[np.where(np.isclose(mexpgrid,14.0,rtol=rtol)),:].ravel(),ls=ls,label=lab+" 10^14 Msol/h")
    # pl.add(zgrid,sngrid[np.where(np.isclose(mexpgrid,14.3,rtol=rtol)),:].ravel(),ls=ls,label=lab+" 10^14.3 Msol/h")
    # pl.add(zgrid,sngrid[np.where(np.isclose(mexpgrid,14.5,rtol=rtol)),:].ravel(),ls=ls,label=lab+" 10^14.5 Msol/h")
    # pl.add(zgrid,sngrid[np.where(np.isclose(mexpgrid,14.7,rtol=rtol)),:].ravel(),ls=ls,label=lab+" 10^14.7 Msol/h")

    # for ind in mindicesList:
    #     pl.add(zgrid,sngrid[ind,:].ravel(),ls=ls,label=lab+" 10^"+str(mexpgrid[ind])+" Msol/h")
    #     print mexpgrid[ind]

    
    plt.gca().set_color_cycle(None)

# pl.legendOn(loc='upper right',labsize=10)
# pl.done(outDir+"slice.pdf")
    
#sys.exit()

#outmgrid = np.arange(min(mmins),max(mmaxes),min(dms))
#outzgrid = np.arange(min(zmins),max(zmaxes),min(dzs))

print outmgrid[0],outmgrid[-1],outmgrid.shape
print outzgrid[0],outzgrid[-1],outzgrid.shape
from orphics.analysis.flatMaps import interpolateGrid

jointgridsqinv = 0.
for key in grids:

    inmgrid,inzgrid,inerrgrid = grids[key]
    outerrgrid = interpolateGrid(inerrgrid,inmgrid,inzgrid,outmgrid,outzgrid,regular=False,kind="cubic",bounds_error=False,fill_value=np.inf)

    mmin = outmgrid[0]
    mmax = outmgrid[-1]
    zmin = outzgrid[0]
    zmax = outzgrid[-1]

    # from orphics.tools.output import Plotter
    # pgrid = np.rot90(1./outerrgrid)
    # pl = Plotter(labelX="$\\mathrm{log}_{10}(M)$",labelY="$z$",ftsize=14)
    # pl.plot2d(pgrid,extent=[mmin,mmax,zmin,zmax],levels=[1.0,3.0,5.0],labsize=14)
    # pl.done(outDir+key+".png")


    jointgridsqinv += (1./outerrgrid**2.)


jointgrid = np.sqrt(1./jointgridsqinv)
snjoint = 1./jointgrid

for ind in mindicesList:
    pl.add(outzgrid,snjoint[ind,:].ravel(),ls="-.")
    print mexpgrid[ind]

pl.legendOn(loc='upper right',labsize=10)
pl.done(outDir+"slice.pdf")



# sngrid = 1./jointgrid
# lab = "joint"
# pl.add(outzgrid,sngrid[np.where(np.isclose(outmgrid,14.0)),:].ravel(),ls=ls,label=lab+" 10^14 Msol/h")
# pl.add(outzgrid,sngrid[np.where(np.isclose(outmgrid,14.3)),:].ravel(),ls=ls,label=lab+" 10^14.3 Msol/h")
# pl.add(outzgrid,sngrid[np.where(np.isclose(outmgrid,14.5)),:].ravel(),ls=ls,label=lab+" 10^14.5 Msol/h")
# pl.add(outzgrid,sngrid[np.where(np.isclose(outmgrid,14.7)),:].ravel(),ls=ls,label=lab+" 10^14.7 Msol/h")

#pl.legendOn(loc='upper right',labsize=8)
#pl.done(outDir+"slice.pdf")


from orphics.tools.io import Plotter
pgrid = np.rot90(1./jointgrid)
pl = Plotter(labelX="$\\mathrm{log}_{10}(M)$",labelY="$z$",ftsize=14)
pl.plot2d(pgrid,extent=[mmin,mmax,zmin,zmax],levels=[1.0,3.0,5.0],labsize=14)
pl.done(outDir+"joint.png")


#savefile = "/astro/astronfs01/workarea/msyriac/data/SZruns/v0.5/lensgrid_S4-1.0-0.4_grid-default_CMB_all_joint_v0.5.pkl"
#savefile = "/astro/astronfs01/workarea/msyriac/data/SZruns/v0.5/lensgridRayUp_S4-1.0-0.4_grid-default_CMB_all_joint_v0.5.pkl"
savefile = "/astro/astronfs01/workarea/msyriac/data/SZruns/v0.5/lensgridRayDn_S4-1.0-0.4_grid-default_CMB_all_joint_v0.5.pkl"
pickle.dump((outmedges,outzedges,jointgrid),open(savefile,'wb'))

# tmgrid = np.arange(mmin,mmax,0.5)
# tzgrid = np.arange(zmin,zmax,0.5)
# coarsegrid = interpolateGrid(jointgrid,outmgrid,outzgrid,tmgrid,tzgrid,regular=False,kind="cubic",bounds_error=False,fill_value=np.inf)

# from orphics.tools.output import Plotter
# pgrid = np.rot90(1./coarsegrid)
# pl = Plotter(labelX="$\\mathrm{log}_{10}(M)$",labelY="$z$",ftsize=14)
# pl.plot2d(pgrid,extent=[mmin,mmax,zmin,zmax],levels=[1.0,3.0,5.0],labsize=14)
# pl.done(outDir+"coarse.png")


# pickle.dump((tmgrid,tzgrid,coarsegrid),open("data/testGrid.pkl",'wb'))
print pgrid.shape
