import szar.sims as s
import orphics.tools.io as io
from configparser import SafeConfigParser
from enlib import enmap,utils,lensing,powspec
import os, sys
import numpy as np
from flipper.fft import fft as fft_gen,ifft as ifft_gen

np.random.seed(100)

def fft(m):
    return fft_gen(m,axes=[-2,-1])
def ifft(m):
    return ifft_gen(m,axes=[-2,-1],normalize=True)

def get_modlmap(imap):
    lmap = imap.lmap()
    return np.sum(lmap**2,0)**0.5



def kappa_to_phi(kappa,modlmap,return_fphi=False):
    fphi = kappa_to_fphi(kappa,modlmap)
    phi =  enmap.samewcs(ifft(fphi).real, kappa) 
    if return_fphi:
        return phi, fphi
    else:
        return phi

def kappa_to_fphi(kappa,modlmap):
    return fkappa_to_fphi(fft(kappa),modlmap)

def fkappa_to_fphi(fkappa,modlmap):
    kmap = np.nan_to_num(2.*fkappa/modlmap/(modlmap+1.))
    kmap[modlmap<2.] = 0.
    return kmap



    


iniFile = "input/params.ini"
#iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)
constDict = io.dictFromSection(Config,'constants')

sim = s.BattagliaSims(constDict)
sim.mapReader(plotRel=True)
sys.exit()

# === TEMPLATE MAP ===
px = 0.05
arc = 20
hwidth = arc/2.
deg = utils.degree
arcmin =  utils.arcmin
shape, wcs = enmap.geometry(pos=[[-hwidth*arcmin,-hwidth*arcmin],[hwidth*arcmin,hwidth*arcmin]], res=px*arcmin, proj="car")


snap = 35
massIndex = 1

maps, z, kappa, szMapuK, projectedM500, trueM500, trueR500, pixScaleX, pixScaleY = sim.getMaps(snap,massIndex,freqGHz=150.,sourceZ=1100.)
print((z, projectedM500, trueM500, trueR500, pixScaleX, pixScaleY))
kappaMap,szMap,projectedM500,z = sim.getKappaSZ(snap,massIndex,shape,wcs,apodWidth=500)

print((projectedM500,z))

print("done")

out_dir = os.environ['WWW']

io.quickPlot2d(kappaMap,out_dir+"kappa.png")#,crange=[-0.1,0.1])
io.highResPlot2d(szMap,out_dir+"sz.png",crange=[-50,0])




B = fft_gen(kappaMap,axes=[-2,-1],flags=['FFTW_MEASURE'])



modlmap = get_modlmap(kappaMap)

io.quickPlot2d(modlmap,out_dir+"modlmap.png")#,crange=[-0.1,0.1])


phi, fphi = kappa_to_phi(kappaMap,modlmap,return_fphi=True)
io.quickPlot2d(phi,out_dir+"phi.png")


alpha_pix = enmap.grad_pixf(fphi)
# alpha_pix2 = enmap.grad_pix(phi)
print((alpha_pix.shape))
# print alpha_pix2.shape

io.quickPlot2d(alpha_pix[0],out_dir+"alpha_pixx.png")
io.quickPlot2d(alpha_pix[1],out_dir+"alpha_pixy.png")
io.quickPlot2d(np.sum(alpha_pix**2,0)**0.5,out_dir+"alpha_pix.png")
# io.quickPlot2d(alpha_pix2[0],out_dir+"alpha_pixx2.png")
# io.quickPlot2d(alpha_pix2[1],out_dir+"alpha_pixy2.png")





TCMB = 2.7255e6
ps = powspec.read_spectrum("../alhazen/data/cl_lensinput.dat")
cmb_map = enmap.rand_map(shape, wcs, ps)/TCMB

lensed = lensing.lens_map_flat_pix(cmb_map, alpha_pix,order=5)
#lensed = lensing.lens_map_flat(cmb_map, phi)

io.quickPlot2d(cmb_map,out_dir+"unlensed.png")
io.quickPlot2d(lensed,out_dir+"lensed.png")
io.quickPlot2d(lensed-cmb_map,out_dir+"diff.png")


alpha = enmap.gradf(fphi)
io.quickPlot2d(alpha[0],out_dir+"alphax.png")
io.quickPlot2d(alpha[1],out_dir+"alphay.png")
io.quickPlot2d(np.sum(alpha**2,0)**0.5,out_dir+"alpha.png")
kappa_inverted = -0.5*enmap.div(alpha,normalize=False)
io.quickPlot2d(kappa_inverted,out_dir+"kappainv.png")

diffper = np.nan_to_num((kappaMap-kappa_inverted)*100./kappa_inverted)
diffper[kappaMap<0.005] = 0.
io.quickPlot2d(diffper,out_dir+"kappadiff.png")


#delensed = lensing.lens_map_flat_pix(lensed, -alpha_pix,order=5)
alpha = enmap.grad(phi)
delensed = lensing.delens_map(lensed, alpha,nstep=5,order=5)
io.quickPlot2d(delensed,out_dir+"delensed.png")
io.quickPlot2d(delensed-cmb_map,out_dir+"delensdiff.png")



