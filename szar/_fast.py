import math
import numpy as np
import numba as nb

    


@nb.jit("f8(f8)",nopython=True)
def m_x(x):
    #NFW cumulative mass distribution
    ans = math.log(1 + x) - x/(1+x)
    return ans

@nb.jit("f8[:](f8[:],f8,f8,f8)",nopython=True)
def rdel_c(M,z,delta,rhocz):
    #spherical overdensity radius w.r.t. the critical density
    ans = (3. * M / (4. * np.pi * delta*rhocz))**(1.0/3.0)
    return ans


@nb.jit("f8(f8,f8,f8,f8)",nopython=True)
def rdel_m(M,z,delta,rhoc0om):
    #spherical overdensity radius w.r.t. the mean matter density
    ans = (3. * M / (4. * np.pi * delta*rhoc0om*(1.+z)**3.))**(1.0/3.0) 
    return ans


@nb.jit("f8(f8,f8)",nopython=True)
def con_M_rel_duffy200(Mvir, z):
    #Duffy 2008 with hs in units MEAN DENSITY 200
    ans = 10.14 / (1. + z)**(1.01) * (Mvir / 2.e12)**(-0.081)
    return ans

@nb.jit("f8(f8,f8)",nopython=True)
def con_M_rel_seljak(Mvir, z):
    #Seljak 2000 with hs in units
    ans = 5.72 / (1 + z) * (Mvir / 10.**14)**(-0.2)
    return ans

@nb.jit("f8(f8,f8)",nopython=True)
def con_M_rel_duffy(Mvir, z):
    #Duffy 2008 with hs in units
    ans = 5.09 / (1 + z)**0.71 * (Mvir / 10.**14)**(-0.081)
    return ans


@nb.jit("f8[:](f8[:],f8,f8,f8,f8,f8)",nopython=True)
def Mass_con_del_2_del_mean200(Mdel,delta,z,rhocz,rhoc0om,ERRTOL):
    #Mass conversion critical to mean overdensity, needed because the Tinker Mass function uses mean matter
    Mass = 2.*Mdel
    rdels = rdel_c(Mdel,z,delta,rhocz)
    ans = Mass*0.0
    for i in range(len(Mdel)):
        while abs(ans[i]/Mass[i] - 1) > ERRTOL : 
            ans[i] = Mass[i]
            conz = con_M_rel_duffy200(Mass[i],z) #DUFFY
            rs = rdel_m(Mass[i],z,200,rhoc0om)/conz
            xx = rdels[i] / rs
            Mass[i] = Mdel[i] * m_x(conz) / m_x(xx)
    ## Finish when they Converge
    return ans
