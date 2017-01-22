import numpy as np

class HaloTools(object):
    def __init__(self,cosmoParams):
        self.cp = cosmoParams

    #NFW cumulative mass distribution
    def m_x(self,x):
        ans = np.log(1 + x) - x/(1+x)
        return ans
    #hubble function
    def E_z(self,z):
        ans = np.sqrt(self.cp.om * (1 + z)**3 + self.cp.ol)
        return ans
    #critical density as a function of z
    def rhoc(self,z):
        ans = self.cp.rho_crit0*self.E_z(z)**2
        return ans

    #spherical overdensity radius w.r.t. the critical density
    def rdel_c(self,M,z,delta):
        ans = (3 * M / (4 * np.pi * delta*self.rhoc(z)))**(1.0/3.0)
        return ans

    #spherical overdensity radius w.r.t. the mean matter density
    def rdel_m(self,M,z,delta):
        ans = (3 * M / (4 * np.pi * delta*self.cp.rho_crit0*self.cp.om*(1.+z)**3))**(1.0/3.0) 
        return ans

    #Seljak 2000 with hs in units
    def con_M_rel_seljak(self,Mvir, z):
        ans = 5.72 / (1 + z) * (Mvir / 10**14)**(-0.2)
        return ans

    #Duffy 2008 with hs in units
    def con_M_rel_duffy(self,Mvir, z):
        ans = 5.09 / (1 + z)**0.71 * (Mvir / 10**14)**(-0.081)
        return ans

    #Duffy 2008 with hs in units MEAN DENSITY 200
    def con_M_rel_duffy200(self,Mvir, z):
        ans = 10.14 / (1 + z)**(1.01) * (Mvir / 2e12)**(-0.081)
        return ans

    #Mass conversion critical to mean overdensity, needed because the Tinker Mass function uses mean matter
    def Mass_con_del_2_del_mean200(self,Mdel,delta,z):
        Mass = 2.*Mdel
        rdels = self.rdel_c(Mdel,z,delta)
        ans = Mass*0.0
        for i in xrange(np.size(Mdel)):
            while np.abs(ans[i]/Mass[i] - 1) > self.cp.c.ERRTOL : 
                ans[i] = Mass[i]
                conz = self.con_M_rel_duffy200(Mass[i],z) #DUFFY
                rs = self.rdel_m(Mass[i],z,200)/conz
                xx = rdels[i] / rs
                Mass[i] = Mdel[i] * self.m_x(conz) / self.m_x(xx)
        ## Finish when they Converge
        return ans
