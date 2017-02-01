import matplotlib
matplotlib.use('Agg')
import numpy as np
from orphics.analysis.flatMaps import interpolateGrid

# beam = 1.0
# noise = 1.0

#mrange = np.arange(14.0,15.4,0.2)
#zrange = np.arange(0.1,0.8,0.2)

saveId = "LAExp_MF_N1"
mrange = np.arange(12.5,15.5,0.05)+0.05
zrange = np.arange(0.05,3.05,0.05)
#mrange = np.arange(14.0,15.7,0.05)
#zrange = np.arange(0.05,2.05,0.05)


mgrid = np.zeros((len(mrange),len(zrange)))

for i,Mexp in enumerate(mrange):
    for j,z in enumerate(zrange):

        #filename =  "data/m"+str(Mexp)+"z"+str(z)+"b"+str(beam)+"n"+str(noise)+".txt"
        #d1,d2,b,n,sn = np.loadtxt(filename,unpack=True)
        filename =  "data/"+saveId+"_m"+str(Mexp)+"_z"+str(z)+".txt"
        try:
            m,z,dlogm = np.loadtxt(filename,unpack=True)
        except:
            print "skipping ", Mexp, z
            dlogm = np.nan
        mgrid[i,j] = dlogm #1./(dlogm*np.sqrt(1000.)) #dlogm
        #print 10**m,z,1./(dlogm)
        #raw_input()

np.savetxt("data/"+saveId+".dat",mgrid)


mfrange = np.arange(12.5,15.5,0.01)+0.01
zfrange = np.arange(0.01,3.01,0.01)
finegrid = interpolateGrid(np.nan_to_num(mgrid),mrange,zrange,mfrange ,zfrange)

#finegrid = mgrid

from orphics.tools.output import Plotter
pl = Plotter()
pl.plot2d(np.rot90(1./finegrid),extent=[12.5,15.5,0.05,3.0],levels=[1.0,2.0,3.0,5.0])
pl.done("output/mgrid.png")
