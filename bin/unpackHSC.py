import numpy as np
from orphics.io import dictFromSection, listFromConfig, Plotter
from orphics.flatMaps import interpolateGrid
import sys, os
from configparser import SafeConfigParser 
import pickle as pickle

calName = sys.argv[1] #"owl1"
gridName = "grid-"+calName
#outDir = os.environ['WWW']

iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)

bigDataDir = Config.get('general','bigDataDirectory')
outDir = bigDataDir
ms = listFromConfig(Config,gridName,'mexprange')
Mexp_edges = np.arange(ms[0],ms[1]+ms[2],ms[2])
zs = listFromConfig(Config,gridName,'zrange')
z_edges = np.arange(zs[0],zs[1]+zs[2],zs[2])


M_edges = 10**Mexp_edges
M = (M_edges[1:]+M_edges[:-1])/2.
mgrid = np.log10(M)

zgrid = (z_edges[1:]+z_edges[:-1])/2.


zz = np.arange(0.1,2.01,0.05)
MMexp = np.arange(13.5,15.71,0.1)
MM = 10**MMexp

hscgrid = np.loadtxt("data/HSC_DeltalnM_z0_z2_17_04_04.txt")


sngrid = 1./hscgrid    
pgrid = np.rot90(sngrid)
pl = Plotter(labelX="$\\mathrm{log}_{10}(M)$",labelY="$z$",ftsize=14)
pl.plot2d(pgrid,extent=[MMexp.min(),MMexp.max(),zz.min(),zz.max()],levels=[3.0,5.0],labsize=14,aspect="auto")
pl.done(outDir+"origHSCgrid.png")


print((hscgrid.shape))


#outmerr = interpolateGrid(hscgrid,MM,zz,M,zgrid,regular=False,kind="cubic",bounds_error=False,fill_value=np.inf)
outmerr = interpolateGrid(hscgrid,MM,zz,M,zgrid,regular=False,kind="cubic",bounds_error=False)


sngrid = 1./outmerr
pgrid = np.rot90(sngrid)
pl = Plotter(labelX="$\\mathrm{log}_{10}(M)$",labelY="$z$",ftsize=14)
pl.plot2d(pgrid,extent=[mgrid.min(),mgrid.max(),zgrid.min(),zgrid.max()],levels=[3.0,5.0],labsize=14,aspect="auto")
pl.done(outDir+"interpHSCgrid.png")

import szar.fisher as sfisher
pickle.dump((Mexp_edges,z_edges,outmerr),open(sfisher.mass_grid_name_owl(bigDataDir,calName),'wb'))
