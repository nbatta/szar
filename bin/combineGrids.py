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


for gridFile in gridList:

    filen = "data/"+gridFile+".pkl"
    
    mexpgrid,zgrid,errgrid = pickle.load(open(filen,'rb'))

    mmin = mexpgrid[0]
    mmax = mexpgrid[-1]
    zmin = zgrid[0]
    zmax = zgrid[-1]

    pgrid = np.rot90(1./errgrid)

    
    from orphics.tools.output import Plotter
    pl = Plotter(labelX="$\\mathrm{log}_{10}(M)$",labelY="$z$",ftsize=14)
    pl.plot2d(pgrid,extent=[mmin,mmax,zmin,zmax],levels=[1.0,3.0,5.0],labsize=14)
    pl.done("output/"+gridFile+".png")
