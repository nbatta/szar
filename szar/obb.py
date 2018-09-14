from __future__ import print_function
from __future__ import division
from builtins import range
from past.utils import old_div
import numpy as np
from scipy.special import spence
from scipy.optimize import fmin
from scipy.optimize import newton
from scipy.integrate import quad
from scipy import special

Gmax = 0.216216538797
delx = 0.01

def nfw(x):
    ans = old_div(1.,(x*(1 + x)**2))
    return ans

def gx(x):
    ans = (np.log(1. + x) - old_div(x,(1. + x)))
    return ans

def gc(c):
    ans = old_div(1.,(np.log(1. + c) - old_div(c,(1. + c))))
    return ans

def Hc(c):
    ans = old_div((-1.*np.log(1 + c)/(1. + c) + c*(1. + 0.5*c)/((1. + c)**2)),gx(c))
    return ans 

def Sc(c):
    ans = (0.5*np.pi**2 - old_div(np.log(c),2.) - old_div(0.5,c) - old_div(0.5,(1 + c)**2) - old_div(3,(1 + c)) +
           np.log(1 + c)*(0.5+old_div(0.5,c**2)-old_div(2,c)-old_div(1,(1+c))) + 
           1.5*(np.log(1 + c))**2 + 3.*spence(c+1))
    
    return ans

def del_s(c):
    ans = old_div(Sc(c), (Sc(c) + (old_div(1.,c**3))*Hc(c)*gx(c)))
    return ans

def K_c(c): #without GMAX
    ans = 1./3.* Hc(c)/(1.-del_s(c))
    return ans

    
def sig_dm2(x,c): ##EQ 14 Lokas & Mamon 2001
    ans = 0.5*x*c*gc(c)*(1 + x)**2 *(np.pi**2 - np.log(x) - (old_div(1.,x)) 
                                     - (old_div(1.,(1. + x)**2)) - (old_div(6.,(1. + x))) 
                                     + np.log(1. + x)*(1. + (old_div(1.,x**2)) - old_div(4.,x) - old_div(2,(1 + x)))
                                    + 3.*(np.log(1. + x))**2 + 6.*spence(x+1))
    return ans


def r200(M):
    z = Params['z']
    om = Params['Omega_m']
    ol = Params['Omega_L']
    Ez2 = om * (1 + z)**3 + ol
    ans = (3 * M / (4 * np.pi * 200.*rhocrit*Ez2))**(old_div(1.0,3.0))
    return ans

def con(Mvir):
    M = old_div(Mvir, Msol_cgs)
    z = Params['z']
    ans = 5.71 / (1 + z)**(0.47) * (old_div(M, 2e12))**(-0.084)
    return ans

def rho_dm(x,Mvir):
    c = con(Mvir)
    rvir = r200(Mvir)
    ans = Mvir*(old_div(c,rvir))**3 / (4.*np.pi*gx(c)) * nfw(x)
    return ans

def jx(x,c):
    ans = 1. - old_div(np.log(1. + x),x)
    ind = np.where(x > c)[0]
    if (len(ind) > 0):
        ans[ind] = 1. -old_div(1.,(1. + c)) - old_div((np.log(1. + c) - old_div(c,(1.+c))),x[ind])
    return ans

def jx_f(x,c):
    if (x <= c):
        ans = 1. - old_div(np.log(1. + x),x)
    else:
        ans = 1. -old_div(1.,(1. + c)) - old_div((np.log(1. + c) - old_div(c,(1.+c))),x)
    return ans

def fx (x,c):
    ans = old_div(np.log(1. + x),x) - old_div(1.,(1. + c))
    ind = np.where(x > c)[0]
    if (len(ind) > 0):
        ans = (old_div(np.log(1. + c),c) - old_div(1.,(1. + c)))*c/x
    return ans

def fstar_func(Mvir):
    ans = 2.5e-2 * (old_div(Mvir, (7e13*Msol_cgs)))**(-0.37) # Modified Giodini 2009 by 0.5
    return ans

def xs_min_func(x,Mvir):
    c = con(Mvir)
    fstar = fstar_func(Mvir)
    ans = gx(c)*fstar/(1. + fstar) - gx(x)
    return ans

def xs_func(Mvir):
    x0 = 1.0
    xs = newton(xs_min_func, x0, args=(Mvir,))
    return xs

def Ks(x_s,Mvir):
    c = con(Mvir)    
    xx = np.arange(old_div(delx,2.),x_s,delx)
    ans = 1./gx(c)*(np.sum(Sc(xx)*xx**2) - 2./3.*np.sum(fx(xx,c)*xx/(1. + xx)**2) )*delx
    return ans

def n_exp(gamma):
    ans = old_div(1., (gamma - 1))
    return ans

def theta_func(x,Mvir,theta,theta2):
    gamma,alpha,Ef = theta
    beta, x_f = theta2
    c = con(Mvir)
    rvir = r200(Mvir)
    nn = n_exp(gamma)
    ans = (1. - beta*jx(x,c)/(1. + nn))
    return ans

def theta_func_f(x,Mvir,theta,theta2):
    gamma,alpha,Ef = theta
    beta, x_f = theta2
    c = con(Mvir)
    rvir = r200(Mvir)
    nn = n_exp(gamma)
    ans = (1. - beta*jx_f(x,c)/(1. + nn))
    return ans

def rho_use(x,Mvir,theta, theta2):
    gamma,alpha,Ef = theta
    beta, x_f = theta2    
    nn = n_exp(gamma)
    ans = (theta_func(x,Mvir,theta,theta2))**nn
    return ans

def rho(x,Mvir,theta, theta2):
    gamma,alpha,Ef = theta
    P_0, rho_0, x_f = theta2  
    nn = n_exp(gamma)
    c = con(Mvir)
    rvir = r200(Mvir)
    beta = rho_0/P_0 * Gravity*Mvir/rvir*c/gx(c)
    theta2_use = beta, x_f
    ans = rho_0*(theta_func(x,Mvir,theta,theta2_use))**nn
    return ans

def rho_outtest(x,Mvir,theta, theta2):
    gamma,alpha,Ef = theta
    P_0, rho_0, x_f = theta2  
    nn = n_exp(gamma)
    c = con(Mvir)
    rvir = r200(Mvir)
    beta = rho_0/P_0 * Gravity*Mvir/rvir*c/gx(c)
    theta2_use = beta, x_f
    print("inside rhoout", theta2_use)
    ans = rho_0*(theta_func(x,Mvir,theta,theta2_use))**nn
    return ans

def Pnth_th(x,Mvir,theta):
    gamma,alpha,Ef = theta
    c = con(Mvir)
    ans = 1. - alpha*(old_div(x,c))**0.8
    return ans

def Pth(x,Mvir,theta,theta2):
    gamma,alpha,Ef = theta
    P_0, rho_0, x_f = theta2
    c = con(Mvir)
    rvir = r200(Mvir)
    nn = n_exp(gamma)
    beta = rho_0/P_0 * Gravity*Mvir/rvir*c/gx(c)
    theta2_use = beta, x_f
    ans = P_0*(theta_func(x,Mvir,theta,theta2_use))**(nn+1.) * Pnth_th(x,Mvir,theta)
    return ans

def Pth_use(x,Mvir,theta,theta2):
    gamma,alpha,Ef = theta
    beta, x_f = theta2    
    nn = n_exp(gamma)
    ans = (theta_func(x,Mvir,theta,theta2))**(nn+1.) * Pnth_th(x,Mvir,theta)
    return ans

def Ptot(Mvir,theta,theta2):
    gamma,alpha,Ef = theta
    P_0, rho_0, x_f = theta2  
    nn = n_exp(gamma)
    beta = rho_0/P_0 * Gravity*Mvir/rvir*c/gx(c)
    theta2_use = beta, x_f    
    ans = P_0*(theta_func_f(x_f,Mvir,theta,theta2_use))**(nn+1.)
    return ans

def Ptot_use(Mvir,theta,theta2):
    gamma,alpha,Ef = theta
    beta, x_f = theta2    
    nn = n_exp(gamma)
    ans = (theta_func_f(x_f,Mvir,theta,theta2))**(nn+1.)
    return ans

def Pnth(x,Mvir,theta,theta2):
    gamma,alpha,Ef = theta
    P_0, rho_0, x_f = theta2
    nn = n_exp(gamma)    
    beta = rho_0/P_0 * Gravity*Mvir/rvir*c/gx(c)
    theta2_use = beta, x_f  
    c = con(Mvir)
    ans = alpha*(old_div(x,c))**0.8 * P_0*(theta_func(x,Mvir,theta,theta2_use))**(nn+1.)
    return ans

def Pnth_use(x,Mvir,theta,theta2):
    gamma,alpha,Ef = theta
    c = con(Mvir)
    nn = n_exp(gamma) 
    ans = alpha*(old_div(x,c))**0.8 * (theta_func(x,Mvir,theta,theta2))**(nn+1.)
    return ans

def I2_int(Mvir,theta,theta2):
    gamma,alpha,Ef = theta
    beta, x_f = theta2    
    nn = n_exp(gamma)
    c = con(Mvir)
    xx = np.arange(old_div(delx,2.),x_f,delx)
    ans = np.sum(fx(xx,c)*rho_use(xx,Mvir,theta,theta2)*xx**2)*delx
    return ans

def I3_int(Mvir,theta,theta2):
    gamma,alpha,Ef = theta
    beta, x_f = theta2    
    nn = n_exp(gamma)
    xx = np.arange(old_div(delx,2.),x_f,delx)
    ans = np.sum(Pth_use(xx,Mvir,theta,theta2) *xx**2)*delx
    return ans

def I4_int(Mvir,theta,theta2):
    gamma,alpha,Ef = theta
    beta, x_f = theta2    
    nn = n_exp(gamma)
    xx = np.arange(old_div(delx,2.),x_f,delx)
    ans = np.sum(Pnth_use(xx,Mvir,theta,theta2)*xx**2)*delx
    return ans

def L_int(Mvir,theta,theta2):
    gamma,alpha,Ef = theta
    beta, x_f = theta2    
    nn = n_exp(gamma)
    xx = np.arange(old_div(delx,2.),x_f,delx)
    ans = np.sum(rho_use(xx,Mvir,theta,theta2)*xx**2)*delx
    return ans

def rho_0_func(theta0,theta2):
    Mvir,gamma,alpha,Ef = theta0
    theta = [gamma,alpha,Ef]
    c = con(Mvir)
    rvir = r200(Mvir)
    fstar = fstar_func(Mvir)
    ans = Mvir*(fb-fstar) / (4.*np.pi * L_int(Mvir,theta,theta2)*(old_div(rvir,c))**3)
    return ans

def P_0_func(theta0,theta2,rho_0):
    Mvir,gamma,alpha,Ef = theta0
    beta, x_f = theta2
    c = con(Mvir)
    rvir = r200(Mvir)
    ans = rho_0/beta * Gravity*Mvir/rvir*c/gx(c)
    return ans

def findroots2(theta2,theta0):
    Mvir,gamma,alpha,Ef = theta0
    theta = [gamma,alpha,Ef]
    beta, x_f = theta2
    c = con(Mvir)
    rvir = r200(Mvir)
    x_s = xs_func(Mvir)
    fstar = fstar_func(Mvir)
    
    E_inj = Ef * gx(c) * rvir * fstar / (Gravity*Mvir*c) * C_CGS**2
    
    Eq1 = (3./2.*(1. + fstar) * (K_c(c)*(3.-4.*del_s(c)) + Ks(x_s,Mvir))  - E_inj + 1./3.* (1.+fstar) *Sc(c) / gx(c) * (x_f**3 - c**3)
           - old_div(I2_int(Mvir,theta,theta2),L_int(Mvir,theta,theta2))
           + 3./2. * I3_int(Mvir,theta,theta2)/(beta*L_int(Mvir,theta,theta2))
           + 3.* I4_int(Mvir,theta,theta2)/(beta*L_int(Mvir,theta,theta2)))
        
    Eq2 = (1.+fstar)*Sc(c) / gx(c) * (beta*L_int(Mvir,theta,theta2)) - Ptot_use(Mvir,theta,theta2)
    
    ans = Eq1**2 + Eq2**2
    return ans

def return_prof_pars(theta2,theta0):
    Mvir,gamma,alpha,Ef = theta0
    beta, x_f = theta2
    ans = fmin(findroots2, theta2, args=(theta0,),disp=False)
    beta_ans, x_f_ans = ans
    rho_0 = rho_0_func(theta0,ans)
    P_0 = P_0_func(theta0,ans,rho_0)
    return P_0, rho_0, x_f_ans 
    
def findroots(theta2,theta0):
    Mvir,gamma,alpha,Ef = theta0
    theta = [gamma,alpha,Ef]
    beta, x_f = theta2
    c = con(Mvir)
    rvir = r200(Mvir)
    E_inj = Ef * gx(c) * rvir / (Gravity*Mvir*c) * C_CGS**2
    
    Eq1 = (3./2. * (K_c(c)*(3.-4.*del_s(c))) - E_inj + 1./3.* Sc(c) / gx(c) * (x_f**3 - c**3)
           - old_div(I2_int(Mvir,theta,theta2),L_int(Mvir,theta,theta2))
           + 3./2. * I3_int(Mvir,theta,theta2)/(beta*L_int(Mvir,theta,theta2))
           + 3.* I4_int(Mvir,theta,theta2)/(beta*L_int(Mvir,theta,theta2)))
    Eq2 = Sc(c) / gx(c) * (beta*L_int(Mvir,theta,theta2)) - Ptot_use(Mvir,theta,theta2)
    return (Eq1,Eq2)
    
def project_prof(tht,Mvir,theta,theta2): # theta2_0

    disc_fac = np.sqrt(2)
    NNR = 100
    drint = 1e-3 * (kpc_cgs * 1e3)
    z = Params['z']
    P0, rho0, x_f = theta2
    fstar = fstar_func(Mvir)
    efac = old_div((2. + 2. * XH), 4.)
    
    rvir = r200(Mvir)/kpc_cgs/1e3
    c = con(Mvir)
    
    sig = 0
    sig2 = 0
    sig_p = 0
    sig2_p = 0
    area_fac = 0
    
    r_ext = AngDist(z)*np.arctan(np.radians(old_div(tht,60.)))
    r_ext2 = AngDist(z)*np.arctan(np.radians(tht*disc_fac/60.))
    
    rad = old_div((np.arange(1e4) + 1.0),1e3) #in MPC
    rad2 = old_div((np.arange(1e4) + 1.0),1e3) #in MPC
    
    radlim = r_ext
    radlim2 = r_ext2
    
    dtht = old_div(np.arctan(old_div(radlim,AngDist(z))),NNR) # rads
    dtht2 = old_div(np.arctan(old_div(radlim2,AngDist(z))),NNR) # rads

    thta = (np.arange(NNR) + 1.)*dtht
    thta2 = (np.arange(NNR) + 1.)*dtht2

    for kk in range(NNR):
        rint = np.sqrt((rad)**2 + thta[kk]**2*AngDist(z)**2)
        rint2 = np.sqrt((rad2)**2 + thta2[kk]**2*AngDist(z)**2)
        
        sig += 2.0*np.pi*dtht*thta[kk]*np.sum(2.*rho(rint/rvir*c,Mvir,theta,theta2)*drint)
        sig2 += 2.0*np.pi*dtht2*thta2[kk]*np.sum(2.*rho(rint2/rvir*c,Mvir,theta,theta2)*drint)

        sig_p += 2.0*np.pi*dtht*thta[kk]*np.sum(2.*Pth(rint/rvir*c,Mvir,theta,theta2)*drint)
        sig2_p += 2.0*np.pi*dtht2*thta2[kk]*np.sum(2.*Pth(rint2/rvir*c,Mvir,theta,theta2)*drint)
        
        area_fac += 2.0*np.pi*dtht*thta[kk]
    
    sig_all =(2*sig - sig2) * 1e-3 * ST_CGS * TCMB * 1e6 * (old_div((2. + 2.*XH),(3.+5.*XH))) / MP_CGS / (np.pi * np.radians(old_div(tht,60.))**2) 
    sig_all_p = (2*sig_p - sig2_p) * ST_CGS/(ME_CGS*C_CGS**2) / area_fac * TCMB * 1e6 * (old_div((2. + 2.*XH),(3.+5.*XH)))# muK

    return sig_all,sig_all_p
