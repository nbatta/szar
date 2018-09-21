import numpy as np
from szar import counts
from orphics import cosmology as cosmo,io
from scipy.interpolate import interp2d
from enlib import bench

class NmzEmulator(object):
    def __init__(self,Mexp_edges,z_edges,cosmo_params=None,const_params=None,low_acc=True):
        cc = counts.ClusterCosmology(cosmo.defaultCosmology if cosmo_params is None else cosmo_params,
                                     constDict=cosmo.defaultConstants if const_params is None else const_params,
                                     lmax=None,skipCls=True,skipPower=False,low_acc=low_acc)
        hmf = counts.Halo_MF(cc,Mexp_edges,z_edges,kh=None,powerZK=None,kmin=1e-4,kmax=5.,knum=200)
        delta = 200
        nmzdensity = hmf.N_of_Mz(hmf.M200,delta)
        Ndz = np.multiply(nmzdensity,np.diff(z_edges).reshape((1,z_edges.size-1)))
        self.Nmz = np.multiply(Ndz,np.diff(10**Mexp_edges).reshape((Mexp_edges.size-1,1))) * 4.* np.pi
        self.Mexp_edges = Mexp_edges
        self.z_edges = z_edges
        self.Medges = 10.**self.Mexp_edges
        self.Mcents = (self.Medges[1:]+self.Medges[:-1])/2.
        self.Mexpcents = np.log10(self.Mcents)
        self.zcents = (self.z_edges[1:]+self.z_edges[:-1])/2.
        self.ntot = self.Nmz.sum()
        self.cc = cc

    def get_catalog(self,poisson=False,seed=None):
        np.random.seed(seed)
        nclusters = int(np.random.poisson(self.ntot)) if poisson else int(self.ntot)
        mzs = np.zeros((nclusters,2),dtype=np.float32)
        print("Generating Nmz catalog...")
        for i in range(nclusters):
            linear_idx = np.random.choice(self.Nmz.size, p=self.Nmz.ravel()/float(self.Nmz.sum()))
            x, y = np.unravel_index(linear_idx, self.Nmz.shape)
            # mzs[i,0] = self.Mexpcents[x]
            # mzs[i,1] = self.zcents[y]
            mzs[i,0] = np.random.uniform(self.Mexpcents[x].min(),self.Mexpcents[x].max())
            mzs[i,1] = np.random.uniform(self.zcents[y].min(),self.zcents[y].max())
        return mzs

def lnlike(nobs,ntheory):
    factorial = lambda x : x*np.log(x) - x
    lnfac = np.log(factorial(nobs))
    return np.nansum(nobs * np.log(ntheory)-ntheory-lnfac)


def main():
    z_edges = np.arange(0.,1.0,0.05)
    Mexp_edges = np.arange(14.0,15.0,0.05)

    emu = NmzEmulator(Mexp_edges,z_edges)
    mzs = emu.get_catalog(poisson=True)
    pdf2d,_,_ = np.histogram2d(mzs[:,0],mzs[:,1],bins=(Mexp_edges,z_edges))
    print (emu.Nmz.sum(),pdf2d.sum(),lnlike(pdf2d,emu.Nmz))


    io.plot_img(pdf2d,"pdf2d.png",flip=False)
    io.plot_img(emu.Nmz,"N2d.png",flip=False)

    data = pdf2d

    true_as = emu.cc.cosmo['As']
    cparams = cosmo.defaultCosmology
    lnlikes = []
    Ases = np.linspace(2.19e-9,2.21e-9,30)
    for As in Ases:
        cparams['As'] = As
        temu = NmzEmulator(Mexp_edges,z_edges,cosmo_params = cparams)
        lnlikes.append(lnlike(data,temu.Nmz))

    lnlikes = np.array(lnlikes)

    pl = io.Plotter(xlabel="As",ylabel="lnlike")
    #pl.add(Ases,np.exp(lnlikes))
    pl.add(Ases,np.exp(lnlikes-lnlikes.max()))
    pl.vline(x=true_as,ls="--")
    pl.done("lnlike.png")

if __name__ == "__main__":
    main()
    
