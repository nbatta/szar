import matplotlib
matplotlib.use('Agg')
import numpy as np

beam = 1.0
noise = 1.0

mrange = np.arange(14.0,15.4,0.2)
zrange = np.arange(0.1,0.8,0.2)

mgrid = np.zeros((len(mrange),len(zrange)))

for i,Mexp in enumerate(mrange):
    for j,z in enumerate(zrange):

        filename =  "output/m"+str(Mexp)+"z"+str(z)+"b"+str(beam)+"n"+str(noise)+".txt"
        d1,d2,b,n,sn = np.loadtxt(filename,unpack=True)
        mgrid[i,j] = sn


from orphics.tools.output import Plotter
pl = Plotter()
pl.plot2d(mgrid)
pl.done("output/mgrid.png")
