from scipy.special import j1
from orphics.theory.quadEstTheory import QuadNorm
import orphics.analysis.flatMaps as fmaps 
import numpy as np
from astLib import astWCS, astCoords
import flipper.liteMap as lm
from orphics.tools.output import Plotter
from orphics.tools.stats import binInAnnuli
from szlib.szcounts import timeit

from scipy.fftpack import fftshift,ifftshift,fftfreq
from pyfftw.interfaces.scipy_fftpack import fft2
from pyfftw.interfaces.scipy_fftpack import ifft2

from scipy.interpolate import splrep,splev

class GRFGen(object):

    def __init__(self,templateLiteMap,ell,Cell,bufferFactor=1,TCMB=2.725e6):
        # Cell is dimensionless

        self.lxMap,self.lyMap,self.modLMap,self.thetaMap,self.lx,self.ly = fmaps.getFTAttributesFromLiteMap(templateLiteMap)


        self.Ny = templateLiteMap.data.shape[1]*bufferFactor
        self.Nx = templateLiteMap.data.shape[0]*bufferFactor

        Ny = self.Ny
        Nx = self.Nx

        bufferFactor = int(bufferFactor)
        self.b = bufferFactor

        realPart = np.zeros([Ny,Nx])
        imgPart  = np.zeros([Ny,Nx])


        s = splrep(ell,Cell*(TCMB**2.),k=3) # maps will be uK fluctuations about zero

        ll = np.ravel(self.modLMap)
        kk = splev(ll,s)
        id = np.where(ll>ell.max())
        kk[id] = 0.

        area = Nx*Ny*templateLiteMap.pixScaleX*templateLiteMap.pixScaleY
        p = np.reshape(kk,[Ny,Nx]) /area * (Nx*Ny)**2
        self.sqp = np.sqrt(p)

    #@timeit
    def getMap(self):
        """
        Generates a GRF from an input power spectrum specified as ell, Cell 
        BufferFactor =1 means the map will be periodic boundary function
        BufferFactor > 1 means the map will be genrated on  a patch bufferFactor times 
        larger in each dimension and then cut out so as to have non-periodic bcs.

        Fills the data field of the map with the GRF realization
        """


        realPart = self.sqp*np.random.randn(self.Ny,self.Nx)
        imgPart = self.sqp*np.random.randn(self.Ny,self.Nx)


        kMap = realPart+1.j*imgPart

        data = np.real(ifft2(kMap)) 

        data = data[(self.b-1)/2*self.Ny:(self.b+1)/2*self.Ny,(self.b-1)/2*self.Nx:(self.b+1)/2*self.Nx]

        return data

# class cmblensCluster(object):

#     def __init__(self,clusterCosmology,M500,z,templateMap,gradCut=None):

#         q = QuadNorm(templateMap,gradCut)


#         data2d = qest.N.Nlkk[polComb]
#         centers, Nlbinned = binInAnnuli(data2d, modLMap, bin_edges)
#         lfilt = stepFunctionFilterFromLiteMap(lm,ellbeam)
#         kapmaker = kappaMaker(Cosmology(defaultLCDM),mass,conc,z,storeKap=False)
#         lm = liteMap.makeEmptyCEATemplate(raSizeDeg=arc/60., decSizeDeg=arc/60.,pixScaleXarcmin = px, pixScaleYarcmin=px)
        
#     def getNoisyStamp(self):
#         kapstamp,kaprad = kapmaker.getKappaAndProfile(Npix,scale=px,beam=None,bin_width=bin_width)
#         lm.fillWithGaussianRandomField(L,Nl/Nsupp,bufferFactor = 1)
#         stamp = lm.data.copy()
#         stamp = stamp+kapstamp.copy()
#         stamp = np.nan_to_num(filterDataFromTemplate(stamp,lfilt))
#         prof = radial_data(stamp,annulus_width=bin_width).mean


#     def sigmaKappa(self,theta500,ells,Nells):



#         integrand = ells*Nells*(theta500*j1(ells*theta500)/ells)**2.
#         return 2.*np.pi.*np.trapz(integrand,ells,np.diff(ells))


#     def avgKappa500(self,M500,z): 
#         pass


#@timeit
def kappa(cc,m500,c500,zL,thetaArc,cmbZ=1100.): #theta in arcminutes


    gnfw = lambda x: np.piecewise(x, [x>1., x<1., x==1.], \
                                [lambda y: (1./(y*y - 1.)) * \
                                 ( 1. - ( (2./np.sqrt(y*y - 1.)) * np.arctan(np.sqrt((y-1.)/(y+1.))) ) ), \
                                 lambda y: (1./(y*y - 1.)) * \
                                ( 1. - ( (2./np.sqrt(-(y*y - 1.))) * np.arctanh(np.sqrt(-((y-1.)/(y+1.)))) ) ), \
                            lambda y: (1./3.)])





    comL  = cc.results.comoving_radial_distance(zL)

    c500  = c500


    comS  = cc.results.comoving_radial_distance(cmbZ)

    comLS = comS-comL


    M = m500
    omegaM = cc.om

    H0 =cc.h * 3.241E-18 #s^-1
    G=4.52E-48 #solar^-1 mpc^3 s^-2
    rhoC0 = 3.*(H0**2.)/(8.*np.pi*G)   #solar / mpc^3

    r500=(3.*M/(4.*np.pi*500.*omegaM*rhoC0))**(1./3.)


    conv=np.pi/(180.*60.)
    theta = thetaArc*conv # theta in radians
    rS = r500/c500

    thetaS = rS/ comL 

    pref=2.*np.pi*1.91E-19 # this is 8piG/c^2 in units of Mpc/solar mass
    fc = np.log(1.+c500) - (c500/(1.+c500))
    const1 = (3./4./np.pi) #dimensionless
    const2 = pref/3. #H0^2 / rhoC0 = 8piG/3/c^2 = pref / 3 in Mpc/solar mass
    const3 = comL * comLS * (1.+zL) / comS #Mpc ############ change back
    const4 = M / (rS*rS) #solar mass / MPc^2
    const5 = 1./fc


    kappaU = gnfw(theta/thetaS)
    consts = const1 * const2 * const3 * const4 * const5
    kappa = consts * kappaU


    return kappa

