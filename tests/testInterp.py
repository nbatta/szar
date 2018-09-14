from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
from orphics.analysis.flatMaps import interpolateGrid
import numpy as np


inM = np.arange(14.0,15.7,0.01)
inZ = np.arange(0.05,2.0,0.01)

print(("inMshape " ,inM.shape))
print(("inZshape " ,inZ.shape))

ZZ,MM = np.meshgrid(inZ,inM)
inGrid = np.exp(-((MM-15.0)**2.+(ZZ-1.0)**2.)/2.)
print(("inGridShape ", inGrid.shape))

from orphics.tools.output import Plotter


pl = Plotter()
pl.plot2d(inGrid)
pl.done("output/ingrid.png")

outM = np.arange(13.5,15.71,0.001)
outZ = np.arange(0.05,2.0,0.001)

print(("outMshape " ,outM.shape))
print(("outZshape " ,outZ.shape))

outGrid = interpolateGrid(inGrid,inM,inZ,outM,outZ)

print(("outGridShape ", outGrid.shape))

pl = Plotter()
pl.plot2d(outGrid)
pl.done("output/outgrid.png")
