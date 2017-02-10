import matplotlib
matplotlib.use('Agg')
import ConfigParser
import numpy as np
import matplotlib.pyplot as plt
import glob
from szlib.szcounts import ClusterCosmology,f_nu
import orphics.analysis.flatMaps as fmaps
import re
from orphics.tools.io import Plotter
            
class BattagliaSims(object):

    def __init__(self,constDict,rootPath="/astro/astronfs01/workarea/msyriac/clusterSims/Battaglia/"):

        cosmoDict = {}
        lmax = 2000
        cosmoDict['H0'] = 72.0
        cosmoDict['om'] = 0.25

        cosmoDict['ob'] = 0.04
        cosmoDict['w0'] = -1
        cosmoDict['mnu'] = 0.

        cosmoDict['ns'] = 0.96
        cosmoDict['As'] = 2.e-9
        

        self.cc = ClusterCosmology(cosmoDict,constDict,lmax)
        self.TCMB = 2.7255e6

        self.root = rootPath

        # Redshifts
        alist = np.loadtxt(rootPath + 'outputSel.txt')
        zlist = np.array((1./alist)-1.)
        snaplist = np.arange(55,55-len(zlist),-1)
        self.snapToZ = lambda s: zlist[snaplist==s][0] 
        self.allSnaps = range(54,34,-1)

        # True masses
        self.NumClust= 300
        self.trueM500overh = {}
        self.trueR500overh = {}
        for snapNum in self.allSnaps:
            snap = str(snapNum)
            f1 = self.root+'cluster_image_info/CLUSTER_MAP_INFO2PBC.L165.256.FBN2_'+snap+'500.d'
            with open(f1, 'rb') as fd:
                temp = np.fromfile(file=fd, dtype=np.float32)
            data = np.reshape(temp,(self.NumClust,4))

            self.trueM500overh[snap] = data[:,2]*1.e10
            self.trueR500overh[snap] = data[:,3]*(1.+self.snapToZ(snapNum))/1.e3 # physical Kpc -> comoving Mpc R500

    def mapReader(self,plotRel=False):
        snapMax = 5
        numMax = 20

        if plotRel:
            pl = Plotter(labelX="true M",labelY="measured M")

        for snap in self.allSnaps[:snapMax]:
            trueMs = []
            expMs = []
            for massIndex in range(self.NumClust)[:numMax]:

                maps, z, totMassMap, szMap, projM500, trueM500, trueR500 = self.getMaps(snap,massIndex)

                print "true M500 " , "{:.2E}".format(trueM500)

                trueMs.append(trueM500)
                expMs.append(projM500)
        
                print "totmass ", "{:.2E}".format(projM500)

                pl = Plotter()
                pl.plot2d(szMap)
                pl.done("output/sz.png")


            if plotRel: pl.add(trueMs,expMs,ls="none",marker="o")#,label=str(snap))

        if plotRel:
            pl.add(trueMs,trueMs,ls="--")
            #pl.legendOn(loc="upper left",labsize=10)
            pl.done("output/dep.png")


    def getMaps(self,snap,massIndex,freqGHz=150.):

        PIX = 2048

        fileStar = self.root + "GEN_Cluster_MassStar_"+str(massIndex)+"L165.256.FBN2_snap"+str(snap)+"_comovFINE.d"
        fileDM = self.root + "GEN_Cluster_MassDM_"+str(massIndex)+"L165.256.FBN2_snap"+str(snap)+"_comovFINE.d"
        fileGas = self.root + "GEN_Cluster_MassGas_"+str(massIndex)+"L165.256.FBN2_snap"+str(snap)+"_comovFINE.d"
        fileY = self.root + "GEN_Cluster_"+str(massIndex)+"L165.256.FBN2_snap"+str(snap)+"_comovFINE.d"

        z = self.snapToZ(snap)

        stampWidthMpc = 8. / self.cc.h
        comovingMpc = self.cc.results.comoving_radial_distance(z)
        widthRadians = stampWidthMpc/comovingMpc
        widthArcmin = widthRadians*60.*180./np.pi
        pixWidthArcmin = widthArcmin/PIX

        areaPixKpc2 = (pixWidthArcmin*comovingMpc*1.e3*np.pi/180./60.)**2.#areaKpc2(pixWidthArcmin)

        maps = {}
        for filen,tag in zip([fileDM,fileStar,fileGas,fileY],["dm","stars","gas","y"]):
            with open(filen, 'rb') as fd:
                temp = np.fromfile(file=fd, dtype=np.float32)

            fac = 1.
            if tag!="y":
                fac = 1.e10*self.cc.h 
    

            #reshape array into 2D array
            map = np.reshape(temp,(PIX,PIX))*fac
            maps[tag] = map.copy()




        totMass = (maps["dm"]+maps["stars"]+maps["gas"])


        class template:
            pass
        t = template()
        t.Nx = PIX
        t.Ny = PIX
        t.pixScaleX = pixWidthArcmin*np.pi/180./60.
        t.pixScaleY = pixWidthArcmin*np.pi/180./60.

        xMap,yMap,modRMap,xx,yy = fmaps.getRealAttributes(t)

        modmapArc = modRMap*60.*180./np.pi


        trueR500 = self.trueR500overh[str(snap)][massIndex]/self.cc.h
        withinArc = trueR500*60.*180./comovingMpc/np.pi
        withinMap = totMass[np.where(modmapArc<withinArc)]
        projectedM500 = withinMap.sum()  *areaPixKpc2 #/ (3.9**2.) # *areaKpc2(arc) check h!!!

        trueM500 = self.trueM500overh[str(snap)][massIndex]/self.cc.h

        freqfac = f_nu(self.cc,freqGHz)
        print freqfac
        szMapuK = maps['y']*freqfac*self.TCMB
        return maps, z, totMass, szMapuK, projectedM500, trueM500, trueR500
