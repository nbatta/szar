import matplotlib
matplotlib.use('Agg')
import sys
import cPickle as pickle
import numpy as np

gridList = sys.argv[1:-1]
outName = sys.argv[-1]

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
for gridFile in gridList:

    filen = "data/"+gridFile+".pkl"
    
    mexpgrid,zgrid,errgrid = pickle.load(open(filen,'rb'))

    mmin = mexpgrid[0]
    mmax = mexpgrid[-1]
    zmin = zgrid[0]
    zmax = zgrid[-1]

    grids[gridFile] = (mexpgrid,zgrid,errgrid)

    mmins.append(mmin)
    mmaxes.append(mmax)
    zmins.append(zmin)
    zmaxes.append(zmax)
    dms.append(min(np.diff(mexpgrid)))
    dzs.append(min(np.diff(zgrid)))

    
    # from orphics.tools.output import Plotter
    # pgrid = np.rot90(1./errgrid)
    # pl = Plotter(labelX="$\\mathrm{log}_{10}(M)$",labelY="$z$",ftsize=14)
    # pl.plot2d(pgrid,extent=[mmin,mmax,zmin,zmax],levels=[1.0,3.0,5.0],labsize=14)
    # pl.done("output/"+gridFile+".png")


outmgrid = np.arange(min(mmins),max(mmaxes),min(dms))
outzgrid = np.arange(min(zmins),max(zmaxes),min(dzs))
from orphics.analysis.flatMaps import interpolateGrid

jointgridsqinv = 0.
for key in grids:

    inmgrid,inzgrid,inerrgrid = grids[key]
    outerrgrid = interpolateGrid(inerrgrid,inmgrid,inzgrid,outmgrid,outzgrid,regular=False,kind="cubic",bounds_error=False,fill_value=np.inf)

    mmin = outmgrid[0]
    mmax = outmgrid[-1]
    zmin = outzgrid[0]
    zmax = outzgrid[-1]

    from orphics.tools.output import Plotter
    pgrid = np.rot90(1./outerrgrid)
    pl = Plotter(labelX="$\\mathrm{log}_{10}(M)$",labelY="$z$",ftsize=14)
    pl.plot2d(pgrid,extent=[mmin,mmax,zmin,zmax],levels=[1.0,3.0,5.0],labsize=14)
    pl.done("output/"+key+".png")


    jointgridsqinv += (1./outerrgrid**2.)


jointgrid = np.sqrt(1./jointgridsqinv)
from orphics.tools.output import Plotter
pgrid = np.rot90(1./jointgrid)
pl = Plotter(labelX="$\\mathrm{log}_{10}(M)$",labelY="$z$",ftsize=14)
pl.plot2d(pgrid,extent=[mmin,mmax,zmin,zmax],levels=[1.0,3.0,5.0],labsize=14)
pl.done("output/joint.png")
