import matplotlib
matplotlib.use('Agg')
import numpy as np
from orphics.tools.output import Plotter
from orphics.analysis.flatMaps import interpolateGrid


saveIdMC = "data/AdvACTCMBLensingWhiteNoise150GhzTTOnly.dat"
saveIdMF = "data/AdvACTCMBLensingWhiteNoise150GhzTTOnly_MF_N1.dat"

mcmat = np.loadtxt(saveIdMC)*np.sqrt(1000.)
mfmat = np.loadtxt(saveIdMF)[:,:-2]

pl = Plotter()
pl.plot2d(mcmat/mfmat)
pl.done("output/ratio.png")

print mcmat/mfmat
print (mcmat/mfmat).mean()

