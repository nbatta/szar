from __future__ import print_function
import configparser
import numpy as np
import matplotlib.pyplot as plt
import glob
from szar.counts import ClusterCosmology
import re
from orphics.io import Plotter
from orphics import maps as fmaps
from enlib import enmap


def f_nu(nu):
    c = {'H_CGS': 6.62608e-27, 'K_CGS': 1.3806488e-16, 'TCMB': 2.7255}
    nu = np.asarray(nu)
    mu = c['H_CGS']*(1e9*nu)/(c['K_CGS']*c['TCMB'])
    ans = mu/np.tanh(mu/2.0) - 4.0
    return ans


class BattagliaReader(object):

    def __init__(self,sim_path="/gpfs01/astro/workarea/msyriac/sims/ClusterSims/"):
        import camb

        self.root = sim_path
        H0 = 72.0
        om = 0.25
        ob = 0.04
        mnu = 0.
        ns = 0.96
        As = 2.e-9
        h = H0/100.
        ombh2 = ob*h**2.
        omh2 = om*h**2.
        omch2 = omh2-ombh2
        self.h = h

        # cosmology for distances
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu)
        pars.InitPower.set_params(ns=ns,As=As)
        self.results= camb.get_background(pars)

        # Snap to  Z
        alist = np.loadtxt(self.root + 'outputSel.txt')
        zlist = np.array((1./alist)-1.)
        snaplist = np.arange(55,55-len(zlist),-1)
        self.snapToZ = lambda s: zlist[snaplist==s][0] 
        self.allSnaps = list(range(54,34,-1))

        # True masses : THIS NEEDS DEBUGGING
        self.NumClust= 300
        self.trueM500 = {}
        self.trueR500 = {}
        for snapNum in self.allSnaps:
            snap = str(snapNum)
            f1 = self.root+'cluster_image_info/CLUSTER_MAP_INFO2PBC.L165.256.FBN2_'+snap+'500.d'
            with open(f1, 'rb') as fd:
                temp = np.fromfile(file=fd, dtype=np.float32)
            data = np.reshape(temp,(self.NumClust,4))

            self.trueM500[snap] = data[:,2]*1.e10 /h # Msun   #/h
            self.trueR500[snap] = data[:,3]*(1.+self.snapToZ(snapNum))/1.e3 #/ self.cc.h # physical Kpc/h -> comoving Mpc R500
        
        # File names

        self.files = {}
        self.files['star'] = lambda massIndex, snap: self.root + "GEN_Cluster_MassStar_"+str(massIndex)+"L165.256.FBN2_snap"+str(snap)+"_comovFINE.d"
        self.files['dm'] = lambda massIndex, snap: self.root + "GEN_Cluster_MassDM_"+str(massIndex)+"L165.256.FBN2_snap"+str(snap)+"_comovFINE.d"
        self.files['gas'] = lambda massIndex, snap: self.root + "GEN_Cluster_MassGas_"+str(massIndex)+"L165.256.FBN2_snap"+str(snap)+"_comovFINE.d"
        self.files['y'] = lambda massIndex, snap: self.root + "GEN_Cluster_"+str(massIndex)+"L165.256.FBN2_snap"+str(snap)+"_comovFINE.d"
        self.files['ksz'] = lambda massIndex, snap: self.root + "GEN_Cluster_kSZ_"+str(massIndex)+"L165.256.FBN2_snap"+str(snap)+"_comovFINE.d"
        self.files['kszN'] = lambda massIndex, snap: self.root + "GEN_Cluster_kSZ_"+str(massIndex)+"L165.256.FBN2_snap"+str(snap)+"MAT.d"

        self.PIX = 2048
        
    def get_geometry(self,snap):
        # MAP DIMENSIONS: THIS NEEDS DEBUGGING
        z = self.snapToZ(snap)
        stampWidthMpc = 8. / self.h
        comovingMpc = self.results.comoving_radial_distance(z)
        widthRadians = stampWidthMpc/comovingMpc
        pixWidthRadians = widthRadians/self.PIX

        shape,wcs = enmap.geometry(pos=np.array([[-widthRadians/2.,-widthRadians/2.],[widthRadians/2.,widthRadians/2.]]),res=pixWidthRadians,proj="car")
        return shape,wcs
    
    def get_map(self,component,mass_index,snap,shape=None,wcs=None):
        if shape is None: 
            assert wcs is None
            shape,wcs = self.get_geometry(snap)
            assert np.all(shape==(self.PIX,self.PIX))
        
        filen = self.files[component](mass_index,snap)
        with open(filen, 'rb') as fd:
            temp = np.fromfile(file=fd, dtype=np.float32)

        dat = np.reshape(temp,shape) * self.h # THIS NEEDS DEBUGGING
        imap = enmap.enmap(dat,wcs)
        return imap

    def get_projected_mass_density(self,mass_index,snap):
        return sum([self.get_map(comp,mass_index,snap) for comp in ['gas','star','dm']]) * 1e10

    def get_kappa(self,mass_index,snap,source_z=1100.):
        z = self.snapToZ(snap)
        comL  = self.results.angular_diameter_distance(z) 
        comS  = self.results.angular_diameter_distance(source_z) 
        comLS = self.results.angular_diameter_distance2(z,source_z) 

        const12 = 2.*9.571e-20 #4G/c^2 in Mpc / solar mass 
        sigmaCr = comS/np.pi/const12/comLS/comL/1e6/(1.+z)**2. # Msolar/kpc2 # NEEDS DEBUGGING!!!
        return self.get_projected_mass_density(mass_index,snap)/sigmaCr

    def get_tsz(self,mass_index,snap,freq_ghz,tcmb=2.7255e6):
        fnu = f_nu(freq_ghz)
        return self.get_compton_y(mass_index,snap)*fnu*tcmb

    def get_compton_y(self,mass_index,snap):
        return self.get_map('y',mass_index,snap)

    def get_ksz(self,mass_index,snap,tcmb=2.7255e6):
        return self.get_map('ksz',mass_index,snap)*tcmb

    def get_ksz_special(self,mass_index,snap,tcmb=2.7255e6):
        shape,wcs = fmaps.rect_geometry(width_arcmin=20.,px_res_arcmin=20./128.)
        return self.get_map('kszN',mass_index,snap,shape=shape,wcs=wcs)*tcmb

    def info(self,mass_index,snap):
        dat = {}
        dat['z'] = self.snapToZ(snap)
        dat['M500'] = self.trueM500[str(snap)][mass_index]
        dat['R500'] = self.trueR500[str(snap)][mass_index]
        shape,wcs = self.get_geometry(snap)
        dat['shape'] = shape
        dat['wcs'] = wcs
        return dat

            
# class BattagliaSims(object):

#     def __init__(self,constDict,rootPath="/gpfs01/astro/workarea/msyriac/sims/ClusterSims/"):

#         # snap 35 to 54
        
#         cosmoDict = {}
#         lmax = 2000
#         cosmoDict['H0'] = 72.0
#         cosmoDict['om'] = 0.25

#         cosmoDict['ob'] = 0.04
#         cosmoDict['w0'] = -1
#         cosmoDict['mnu'] = 0.

#         cosmoDict['ns'] = 0.96
#         cosmoDict['As'] = 2.e-9
        

#         self.cc = ClusterCosmology(cosmoDict,constDict.copy(),lmax,pickling=True,skipCls=True,verbose=False)
#         self.TCMB = 2.7255e6

#         self.root = rootPath

#         # Redshifts
#         alist = np.loadtxt(rootPath + 'outputSel.txt')
#         zlist = np.array((1./alist)-1.)
#         snaplist = np.arange(55,55-len(zlist),-1)
#         self.snapToZ = lambda s: zlist[snaplist==s][0] 
#         self.allSnaps = list(range(54,34,-1))

#         # True masses
#         self.NumClust= 300
#         self.trueM500 = {}
#         self.trueR500 = {}
#         for snapNum in self.allSnaps:
#             snap = str(snapNum)
#             f1 = self.root+'cluster_image_info/CLUSTER_MAP_INFO2PBC.L165.256.FBN2_'+snap+'500.d'
#             with open(f1, 'rb') as fd:
#                 temp = np.fromfile(file=fd, dtype=np.float32)
#             data = np.reshape(temp,(self.NumClust,4))

#             self.trueM500[snap] = data[:,2]*1.e10 /self.cc.h # Msun   #/h
#             self.trueR500[snap] = data[:,3]*(1.+self.snapToZ(snapNum))/1.e3 #/ self.cc.h # physical Kpc/h -> comoving Mpc R500



#     def mapReader(self,plotRel=False):
#         snapMax = 5
#         numMax = 20

#         if plotRel:
#             pl = Plotter(labelX="true M",labelY="measured M")

#         for snap in self.allSnaps[:snapMax]:
#             trueMs = []
#             expMs = []
#             for massIndex in range(self.NumClust)[:numMax]:

#                 maps, z, kappa, szMap, projM500, trueM500, trueR500, pixScaleX, pixScaleY = self.getMaps(snap,massIndex)

#                 print("true M500 " , "{:.2E}".format(trueM500))

#                 trueMs.append(trueM500)
#                 expMs.append(projM500)
        
#                 print("totmass ", "{:.2E}".format(projM500))

#                 if not(plotRel):
#                     pl = Plotter()
#                     pl.plot2d(szMap)
#                     pl.done("output/sz.png")


#             if plotRel: pl.add(trueMs,expMs,ls="none",marker="o")#,label=str(snap))

#         if plotRel:
#             pl.add(trueMs,trueMs,ls="--")
#             #pl.legendOn(loc="upper left",labsize=10)
#             pl.done("output/dep.png")


#     def getMaps(self,snap,massIndex,freqGHz=150.,sourceZ=1100.):

#         PIX = 2048

#         fileStar = self.root + "GEN_Cluster_MassStar_"+str(massIndex)+"L165.256.FBN2_snap"+str(snap)+"_comovFINE.d"
#         fileDM = self.root + "GEN_Cluster_MassDM_"+str(massIndex)+"L165.256.FBN2_snap"+str(snap)+"_comovFINE.d"
#         fileGas = self.root + "GEN_Cluster_MassGas_"+str(massIndex)+"L165.256.FBN2_snap"+str(snap)+"_comovFINE.d"
#         fileY = self.root + "GEN_Cluster_"+str(massIndex)+"L165.256.FBN2_snap"+str(snap)+"_comovFINE.d"

#         z = self.snapToZ(snap)
#         #print("Snap ", snap , " corresponds to redshift ", z)

#         stampWidthMpc = 8. / self.cc.h
#         comovingMpc = self.cc.results.comoving_radial_distance(z)
#         widthRadians = stampWidthMpc/comovingMpc
#         widthArcmin = widthRadians*60.*180./np.pi
#         pixWidthArcmin = widthArcmin/PIX

#         areaPixKpc2 = (pixWidthArcmin*comovingMpc*1.e3*np.pi/180./60.)**2.#areaKpc2(pixWidthArcmin)
#         # areaPixKpc2_test = (stampWidthMpc/PIX*1.e3)**2.#areaKpc2(pixWidthArcmin)
#         # print areaPixKpc2, areaPixKpc2_test, np.sqrt(areaPixKpc2)*self.cc.h

#         # sys.exit()

#         maps = {}
#         for filen,tag in zip([fileDM,fileStar,fileGas,fileY],["dm","stars","gas","y"]):
#             with open(filen, 'rb') as fd:
#                 temp = np.fromfile(file=fd, dtype=np.float32)

#             fac = self.cc.h # 1.
#             if tag!="y":
#                 fac = 1.e10*self.cc.h 
    

#             #reshape array into 2D array
#             map = np.reshape(temp,(PIX,PIX))*fac
#             maps[tag] = map.copy()




#         totMass = (maps["dm"]+maps["stars"]+maps["gas"])


#         # class template:
#         #     pass
#         # t = template()
#         # t.Nx = PIX
#         # t.Ny = PIX
#         pixScaleX = pixWidthArcmin*np.pi/180./60.
#         pixScaleY = pixWidthArcmin*np.pi/180./60.

        
#         xMap,yMap,modRMap,xx,yy = fmaps.get_real_attributes_generic(PIX,PIX,pixScaleY,pixScaleX)

#         modmapArc = modRMap*60.*180./np.pi


#         trueR500 = self.trueR500[str(snap)][massIndex] #/self.cc.h
#         withinArc = trueR500*60.*180./comovingMpc/np.pi
#         withinMap = totMass[np.where(modmapArc<withinArc)]
#         projectedM500 = withinMap.sum()  *areaPixKpc2 #/ (3.9**2.) # *areaKpc2(arc) check h!!!

#         trueM500 = self.trueM500[str(snap)][massIndex] #/self.cc.h

#         #assert projectedM500>trueM500

#         freqfac = f_nu(self.cc.c,freqGHz)
#         # print(freqfac)
#         szMapuK = maps['y']*freqfac*self.TCMB

#         # cmbZ = sourceZ
#         # comL  = self.cc.results.comoving_radial_distance(z) 
#         # comS  = self.cc.results.comoving_radial_distance(cmbZ) 
#         # comLS = comS-comL

#         # const12 = 9.571e-20 # 2G/c^2 in Mpc / solar mass 
#         # sigmaCr = comS/np.pi/const12/comLS/comL/1e6 # Msolar/kpc2


#         cmbZ = sourceZ
#         comL  = self.cc.results.angular_diameter_distance(z) 
#         comS  = self.cc.results.angular_diameter_distance(cmbZ) 
#         comLS = self.cc.results.angular_diameter_distance2(z,cmbZ) 

#         const12 = 2.*9.571e-20 #4G/c^2 in Mpc / solar mass 
#         sigmaCr = comS/np.pi/const12/comLS/comL/1e6/(1.+z)**2. # Msolar/kpc2



#         kappa = totMass/sigmaCr # not sure if totMass needs to be rescaled?

#         return maps, z, kappa, szMapuK, projectedM500, trueM500, trueR500, pixScaleX, pixScaleY

    
#     def getKappaSZ(self,snap,massIndex,shape,wcs,apodWidthArcmin=None):

#         from enlib import enmap,utils
#         arcmin =  utils.arcmin
#         PIX = 2048
#         maps, z, kappaSimDat, szMapuKDat, projectedM500, trueM500, trueR500, pxInRad, pxInRad = self.getMaps(snap,massIndex,freqGHz=150.)
#         pxIn = pxInRad * 180.*60./np.pi
#         hwidth = PIX*pxIn/2.

#         # input pixelization
#         shapeSim, wcsSim = enmap.geometry(pos=[[-hwidth*arcmin,-hwidth*arcmin],[hwidth*arcmin,hwidth*arcmin]], res=pxIn*arcmin, proj="car")
#         kappaMap = enmap.enmap(kappaSimDat,wcsSim)
#         szMap = enmap.enmap(szMapuKDat,wcsSim)


#         if apodWidthArcmin is not None:
#             apodWidth = int(apodWidthArcmin/pxIn)
#             kappaMap = enmap.apod(kappaMap,apodWidth)
#             szMap = enmap.apod(szMap,apodWidth)

#         # from orphics import io
#         # io.plot_img(kappaMap,io.dout_dir+"battaglia_cluster_kappa_"+str(massIndex)+".png",verbose=False)
#         # io.plot_img(szMap,io.dout_dir+"battaglia_cluster_tsz_"+str(massIndex)+".png",verbose=False)

            
#         kappaMap = enmap.project(kappaMap, shape, wcs)
#         szMap = enmap.project(szMap, shape, wcs)
        
#         assert szMap.shape==shape
#         assert kappaMap.shape==shape

#         # print "kappaint ", kappaMap[thetaMap*60.*180./np.pi<10.].mean()
#         return kappaMap,szMap,projectedM500,z
