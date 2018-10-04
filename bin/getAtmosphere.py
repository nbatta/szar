from __future__ import print_function
from orphics.cosmology import getAtmosphere
import numpy as np
beamList = np.arange(1.0,3.5,0.5)


def s(num):
    return '{:.1f}'.format(float(num)).strip()

for beamFWHMArcmin in beamList:
    ttl,tta,ppl,ppa= getAtmosphere(beamFWHMArcmin,returnFunctions=False)
    print(beamFWHMArcmin)
    print(s(ttl)+","+s(ppl))
    print(s(tta)+","+s(ppa))
