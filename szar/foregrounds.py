import numpy as np
from scipy.interpolate import interp1d

def f_nu(constDict,nu):
    c = constDict
    mu = c['H_CGS']*(1e9*nu)/(c['K_CGS']*c['TCMB'])
    ans = mu/np.tanh(mu/2.0) - 4.0
    return ans

def g_nu(constDict,nu):
    c = constDict
    beta = (nu*1e9) * c['H_CGS'] / (c['K_CGS']*c['TCMB'])
    ans = 2.* c['H_CGS']**2 * (nu*1e9)**4 / (c['C']**2 *c['K_CGS'] * c['TCMB']**2) \
        * np.exp(beta) * 1. / (np.exp(beta) - 1.)**2
    return ans


def totTTNoise(ells,constDict,beamFWHM,noiseT,freq,lknee,alpha,tsz_battaglia_template_csv="data/sz_template_battaglia.csv",TCMB=2.7255e6):
    ls = ells
    instrument = noise_func(ls,beamFWHM,noiseT,lknee,alpha,dimensionless=False)/ cc.c['TCMBmuK']**2.
    fgs = fgNoises(constDict,tsz_battaglia_template_csv)
    ksz = fgs.ksz_temp(ls)/ls/(ls+1.)*2.*np.pi/ cc.c['TCMBmuK']**2.
    radio = fgs.rad_ps(ls,freq,freq)/ls/(ls+1.)*2.*np.pi/ cc.c['TCMBmuK']**2.
    cibp = fgs.cib_p(ls,freq,freq) /ls/(ls+1.)*2.*np.pi/ cc.c['TCMBmuK']**2.
    cibc = fgs.cib_c(ls,freq,freq)/ls/(ls+1.)*2.*np.pi/ cc.c['TCMBmuK']**2.
    tsz = fgs.tSZ(ls,freq,freq)/ls/(ls+1.)*2.*np.pi/ cc.c['TCMBmuK']**2.
    tsz_cib = fgs.tSZ_CIB(ls,freq,freq)/ls/(ls+1.)*2.*np.pi/ cc.c['TCMBmuK']**2.
    return instrument+ksz+radio+cibp+cibc+tsz #+tsz_cib



class fgNoises(object):
    '''                                                                                                                             
    Returns fgPower * l(l+1)/2pi in uK^2                                                                                            
    '''

    def __init__(self,constDict,ksz_file='input/ksz_BBPS.txt',ksz_p_file='input/ksz_p_BBPS.txt',tsz_cib_file='input/sz_x_cib_template.dat',ksz_battaglia_test_csv=None,tsz_battaglia_template_csv=None,components=None,lmax=None):
        self.c = constDict
        el,ksz = np.loadtxt(ksz_file,unpack=True)
        self.ksz_func = interp1d(el,ksz,bounds_error=False,fill_value=0.)
        elp,kszp = np.loadtxt(ksz_p_file,unpack=True)
        self.ksz_p_func = interp1d(elp,kszp,bounds_error=False,fill_value=0.)
        eltc,tsz_cib = np.loadtxt(tsz_cib_file,unpack=True)
        self.tsz_cib_func = interp1d(eltc,tsz_cib,bounds_error=False,fill_value=0.)

        if ksz_battaglia_test_csv is not None:
            ells,cls = np.loadtxt(ksz_battaglia_test_csv,delimiter=',',unpack=True)
            clfunc = interp1d(ells,cls,bounds_error=False,fill_value=0.)
            self.ksz_battaglia_test = lambda ell: 1.65*clfunc(ell)

        if tsz_battaglia_template_csv is not None:
            ells,cls = np.loadtxt(tsz_battaglia_template_csv,delimiter=',',unpack=True)
            self.tsz_template = interp1d(ells,cls,bounds_error=False,fill_value=0.)
        else:
            self.tsz_template = None

        if components is not None:
            fgdict = {'tsz':self.tSZ,'cibc':self.cib_c,'cibp':self.cib_p,'radps':self.rad_ps}
            self.fgdict_nu = {'tsz':self.tSZ_nu,'cibc':self.cib_nu,'cibp':self.cib_nu,'radps':self.rad_ps_nu}
            self.ells = np.arange(0,lmax,1)
            self.nu0 = 150.
            self.noises = {}
            for component in components:
                fgfunc = fgdict[component]
                noise = fgfunc(self.ells,self.nu0,self.nu0)*2.*np.pi*np.nan_to_num(1./self.ells/(self.ells+1.))
                self.noises[component] = interp1d(self.ells,noise,bounds_error=False,fill_value=0.)
            

    def get_noise(self,component,nu1,nu2,ells):
        return self.noises[component](ells)* self.fgdict_nu[component](nu1)*self.fgdict_nu[component](nu2)/self.fgdict_nu[component](self.nu0)**2.
    
    def g_nu(self,nu):
        beta = (nu*1e9) * self.c['H_CGS'] / (self.c['K_CGS']*self.c['TCMB'])
        ans = 2.* self.c['H_CGS']**2 * (nu*1e9)**4 / (self.c['C']**2 *self.c['K_CGS'] * self.c['TCMB']**2) \
            * np.exp(beta) * 1. / (np.exp(beta) - 1.)**2
        return ans

    def B_nu(self,Td,nu):
        beta = (nu*1e9) * self.c['H_CGS'] / (self.c['K_CGS']*Td)
        ans = 2.* self.c['H_CGS'] * nu**3 / (self.c['C'])**2 / (np.exp(beta) - 1.)
        return ans

    def rad_ps_nu(self,nu):
        return (nu/self.c['nu0']) ** self.c['al_ps'] \
            * self.g_nu(nu)  / (self.g_nu(self.c['nu0']))
        
    def rad_ps(self,ell,nu1,nu2):
        ans = self.c['A_ps'] * (ell/self.c['ell0sec']) ** 2 * self.rad_ps_nu(nu1)*self.rad_ps_nu(nu2)
        return ans

    def res_gal_nu(self,nu):
        return (nu/self.c['nu0']) ** self.c['al_g'] \
            * self.g_nu(nu)  / (self.g_nu(self.c['nu0']))
    
    def res_gal(self,ell,nu1,nu2):
        ans = self.c['A_g'] * (ell/self.c['ell0sec']) ** self.c['n_g'] * self.res_gal_nu(nu1)*self.res_gal_nu(nu2)
        return ans

    def cib_nu(self,nu):

        mu = nu**self.c['al_cib']*self.B_nu(self.c['Td'],nu) * self.g_nu(nu)
        mu0 = self.c['nu0']**self.c['al_cib']*self.B_nu(self.c['Td'],self.c['nu0']) \
            * self.g_nu(self.c['nu0'])
        return mu  / mu0

    def cib_p(self,ell,nu1,nu2):
        
        ans = self.c['A_cibp'] * (ell/self.c['ell0sec']) ** 2.0 * self.cib_nu(nu1)*self.cib_nu(nu2) 
        return ans

    def cib_c(self,ell,nu1,nu2):

        ans = self.c['A_cibc'] * (ell/self.c['ell0sec']) ** (2.-self.c['n_cib']) * self.cib_nu(nu1)*self.cib_nu(nu2) 
        return ans

    def f_nu_cib(self,nu1):
        return self.cib_nu(nu1)
        # mu1 = nu1**self.c['al_cib']*self.B_nu(self.c['Td'],nu1) * self.g_nu(nu1)
        # mu0 = self.c['nu0']**self.c['al_cib']*self.B_nu(self.c['Td'],self.c['nu0']) \
        #     * self.g_nu(self.c['nu0'])
        # ans = mu1 /  mu0

        # return ans

    def tSZ_CIB_nu(self,nu1,nu2):
        mu1 = nu1**self.c['al_cib']*self.B_nu(self.c['Td'],nu1) * self.g_nu(nu1)
        mu2 = nu2**self.c['al_cib']*self.B_nu(self.c['Td'],nu2) * self.g_nu(nu2)
        mu0 = self.c['nu0']**self.c['al_cib']*self.B_nu(self.c['Td'],self.c['nu0']) \
            * self.g_nu(self.c['nu0'])

        fp12 = f_nu(self.c,nu1)*mu1 + f_nu(self.c,nu2)*mu2
        fp0  = 2.* f_nu(self.c,self.c['nu0'])*mu0
        ans = self.c['zeta']*np.sqrt(self.c['A_tsz']*self.c['A_cibc']) * 2.*fp12 / fp0 
        return ans

    def tSZ_CIB(self,ell,nu1,nu2):
        ans = self.tSZ_CIB_nu(nu1,nu2) * self.tsz_cib_func(ell)
        return ans

    def ksz_temp(self,ell):
        ans = self.ksz_func(ell) * (1.65/1.5) + self.ksz_p_func(ell)
        return ans

    def tSZ_nu(self,nu):
        return f_nu(self.c,nu)/f_nu(self.c,self.c['nu0'])
    
    def tSZ(self,ell,nu1,nu2):
        assert self.tsz_template is not None, "You did not initialize this object with tsz_battaglia_template_csv."
        return self.c['A_tsz']*self.tsz_template(ell)*self.tSZ_nu(nu1)*self.tSZ_nu(nu2)

    def gal_dust_pol(self,ell,nu1,nu2):
        mu1 = nu1**self.c['al_cib']*self.B_nu(self.c['Td'],nu1) * self.g_nu(nu1)
        mu2 = nu2**self.c['al_cib']*self.B_nu(self.c['Td'],nu2) * self.g_nu(nu2)
        mu0 = self.c['nu0']**self.c['al_cib']*self.B_nu(self.c['Td'],self.c['nu0']) \
            * self.g_nu(self.c['nu0'])

        ans = self.c['A_gal_dust'] * (ell/self.c['ell0sec']) ** self.c['alpha_gd'] * mu1 * mu2 / mu0**2
        return ans

    def gal_sync_pol(self,ell,nu1,nu2):
        ans = self.c['A_gal_sync'] * (ell/self.c['ell0sec']) ** self.c['alpha_gs'] \
            * (nu1*nu2/self.c['nu0']**2) ** self.c['al_ps'] * self.g_nu(nu1) * self.g_nu(nu2) / (self.g_nu(self.c['nu0']))**2
        return ans

    def rad_pol_ps(self,ell,nu1,nu2):
        ans = self.c['A_ps_pol'] * (ell/self.c['ell0sec']) ** 2 * (nu1*nu2/self.c['nu0']**2) ** self.c['al_ps'] \
            * self.g_nu(nu1) * self.g_nu(nu2) / (self.g_nu(self.c['nu0']))**2
        return ans



class fgGenerator(fgNoises):
    
    def __init__(self,shape,wcs,components,constDict,ksz_file='input/ksz_BBPS.txt',ksz_p_file='input/ksz_p_BBPS.txt',tsz_cib_file='input/sz_x_cib_template.dat',ksz_battaglia_test_csv=None,tsz_battaglia_template_csv=None):
        
        from enlib import enmap
        from orphics import maps

        modlmap = enmap.modlmap(shape,wcs)
        fgNoises.__init__(self,constDict,ksz_file,ksz_p_file,tsz_cib_file,ksz_battaglia_test_csv,tsz_battaglia_template_csv,components,lmax=modlmap.max())
        
        self.mgens = {}
        for component in components:
            noise = self.noises[component](self.ells)
            ps = noise.reshape((1,1,self.ells.size))
            self.mgens[component] = maps.MapGen(shape,wcs,ps)

    def get_maps(self,component,nus,seed=None):
        rmap = self.mgens[component].get_map(seed=seed)
        return [rmap * self.fgdict_nu[component](nu)/self.fgdict_nu[component](self.nu0) for nu in nus]

        
    
