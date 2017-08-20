import matplotlib
matplotlib.use('Agg')
import ConfigParser
import numpy as np
import matplotlib.pyplot as plt
import glob
from szar.counts import ClusterCosmology
from szar.foregrounds import f_nu
import orphics.analysis.flatMaps as fmaps
import re
from orphics.tools.io import Plotter
            
class BattagliaSims(object):

    def __init__(self,constDict,rootPath="/gpfs01/astro/workarea/msyriac/sims/ClusterSims/"):

        # snap 35 to 54
        
        cosmoDict = {}
        lmax = 2000
        cosmoDict['H0'] = 72.0
        cosmoDict['om'] = 0.25

        cosmoDict['ob'] = 0.04
        cosmoDict['w0'] = -1
        cosmoDict['mnu'] = 0.

        cosmoDict['ns'] = 0.96
        cosmoDict['As'] = 2.e-9
        

        self.cc = ClusterCosmology(cosmoDict,constDict,lmax,pickling=True)
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
        self.trueM500 = {}
        self.trueR500 = {}
        for snapNum in self.allSnaps:
            snap = str(snapNum)
            f1 = self.root+'cluster_image_info/CLUSTER_MAP_INFO2PBC.L165.256.FBN2_'+snap+'500.d'
            with open(f1, 'rb') as fd:
                temp = np.fromfile(file=fd, dtype=np.float32)
            data = np.reshape(temp,(self.NumClust,4))

            self.trueM500[snap] = data[:,2]*1.e10 /self.cc.h # Msun   #/h
            self.trueR500[snap] = data[:,3]*(1.+self.snapToZ(snapNum))/1.e3 #/ self.cc.h # physical Kpc/h -> comoving Mpc R500

    def mapReader(self,plotRel=False):
        snapMax = 5
        numMax = 20

        if plotRel:
            pl = Plotter(labelX="true M",labelY="measured M")

        for snap in self.allSnaps[:snapMax]:
            trueMs = []
            expMs = []
            for massIndex in range(self.NumClust)[:numMax]:

                maps, z, kappa, szMap, projM500, trueM500, trueR500, pixScaleX, pixScaleY = self.getMaps(snap,massIndex)

                print "true M500 " , "{:.2E}".format(trueM500)

                trueMs.append(trueM500)
                expMs.append(projM500)
        
                print "totmass ", "{:.2E}".format(projM500)

                if not(plotRel):
                    pl = Plotter()
                    pl.plot2d(szMap)
                    pl.done("output/sz.png")


            if plotRel: pl.add(trueMs,expMs,ls="none",marker="o")#,label=str(snap))

        if plotRel:
            pl.add(trueMs,trueMs,ls="--")
            #pl.legendOn(loc="upper left",labsize=10)
            pl.done("output/dep.png")


    def getMaps(self,snap,massIndex,freqGHz=150.,sourceZ=1100.):

        PIX = 2048

        fileStar = self.root + "GEN_Cluster_MassStar_"+str(massIndex)+"L165.256.FBN2_snap"+str(snap)+"_comovFINE.d"
        fileDM = self.root + "GEN_Cluster_MassDM_"+str(massIndex)+"L165.256.FBN2_snap"+str(snap)+"_comovFINE.d"
        fileGas = self.root + "GEN_Cluster_MassGas_"+str(massIndex)+"L165.256.FBN2_snap"+str(snap)+"_comovFINE.d"
        fileY = self.root + "GEN_Cluster_"+str(massIndex)+"L165.256.FBN2_snap"+str(snap)+"_comovFINE.d"

        z = self.snapToZ(snap)
        print "Snap ", snap , " corresponds to redshift ", z

        stampWidthMpc = 8. / self.cc.h
        comovingMpc = self.cc.results.comoving_radial_distance(z)
        widthRadians = stampWidthMpc/comovingMpc
        widthArcmin = widthRadians*60.*180./np.pi
        pixWidthArcmin = widthArcmin/PIX

        areaPixKpc2 = (pixWidthArcmin*comovingMpc*1.e3*np.pi/180./60.)**2.#areaKpc2(pixWidthArcmin)
        # areaPixKpc2_test = (stampWidthMpc/PIX*1.e3)**2.#areaKpc2(pixWidthArcmin)
        # print areaPixKpc2, areaPixKpc2_test, np.sqrt(areaPixKpc2)*self.cc.h

        # sys.exit()

        maps = {}
        for filen,tag in zip([fileDM,fileStar,fileGas,fileY],["dm","stars","gas","y"]):
            with open(filen, 'rb') as fd:
                temp = np.fromfile(file=fd, dtype=np.float32)

            fac = self.cc.h # 1.
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


        trueR500 = self.trueR500[str(snap)][massIndex] #/self.cc.h
        withinArc = trueR500*60.*180./comovingMpc/np.pi
        withinMap = totMass[np.where(modmapArc<withinArc)]
        projectedM500 = withinMap.sum()  *areaPixKpc2 #/ (3.9**2.) # *areaKpc2(arc) check h!!!

        trueM500 = self.trueM500[str(snap)][massIndex] #/self.cc.h

        #assert projectedM500>trueM500

        freqfac = f_nu(self.cc.c,freqGHz)
        print freqfac
        szMapuK = maps['y']*freqfac*self.TCMB

        # cmbZ = sourceZ
        # comL  = self.cc.results.comoving_radial_distance(z) 
        # comS  = self.cc.results.comoving_radial_distance(cmbZ) 
        # comLS = comS-comL

        # const12 = 9.571e-20 # 2G/c^2 in Mpc / solar mass 
        # sigmaCr = comS/np.pi/const12/comLS/comL/1e6 # Msolar/kpc2


        cmbZ = sourceZ
        comL  = self.cc.results.angular_diameter_distance(z) 
        comS  = self.cc.results.angular_diameter_distance(cmbZ) 
        comLS = self.cc.results.angular_diameter_distance2(z,cmbZ) 

        const12 = 2.*9.571e-20 #4G/c^2 in Mpc / solar mass 
        sigmaCr = comS/np.pi/const12/comLS/comL/1e6/(1.+z)**2. # Msolar/kpc2



        kappa = totMass/sigmaCr # not sure if totMass needs to be rescaled?

        return maps, z, kappa, szMapuK, projectedM500, trueM500, trueR500, t.pixScaleX, t.pixScaleY

    
    def getKappaSZ(self,snap,massIndex,shape,wcs,apodWidthArcmin=None):

        from enlib import enmap,utils
        arcmin =  utils.arcmin
        PIX = 2048
        maps, z, kappaSimDat, szMapuKDat, projectedM500, trueM500, trueR500, pxInRad, pxInRad = self.getMaps(snap,massIndex,freqGHz=150.)
        pxIn = pxInRad * 180.*60./np.pi
        hwidth = PIX*pxIn/2.

        # input pixelization
        shapeSim, wcsSim = enmap.geometry(pos=[[-hwidth*arcmin,-hwidth*arcmin],[hwidth*arcmin,hwidth*arcmin]], res=pxIn*arcmin, proj="car")
        kappaMap = enmap.enmap(kappaSimDat,wcsSim)
        szMap = enmap.enmap(szMapuKDat,wcsSim)


        if apodWidthArcmin is not None:
            apodWidth = int(apodWidthArcmin/pxIn)
            kappaMap = enmap.apod(kappaMap,apodWidth)
            szMap = enmap.apod(szMap,apodWidth)


        kappaMap = enmap.project(kappaMap, shape, wcs)
        szMap = enmap.project(szMap, shape, wcs)
        
        assert szMap.shape==shape
        assert kappaMap.shape==shape

        # print "kappaint ", kappaMap[thetaMap*60.*180./np.pi<10.].mean()
        return kappaMap,szMap,projectedM500,z
