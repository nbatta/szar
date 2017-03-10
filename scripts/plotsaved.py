import sys
import numpy as np
from orphics.tools.io import Plotter

def cov2corr(cov):
    d = np.diag(cov)
    stddev = np.sqrt(d)
    corr = cov.copy()*0.
    for i in range(cov.shape[0]):
        for j in range(cov.shape[0]):
            corr[i,j] = cov[i,j]/stddev[i]/stddev[j]

    return corr


#zout = np.arange(0.05,3.0,0.1)
zout = np.arange(0.05,2.0,0.1)

planckFile = "data/Feb26_FisherMat_Planck_tau0.01_lens_fsky0.6_lcdm.csv"
fisherPlanck = np.loadtxt(planckFile)
numLeft = zout.size+7
fisherPlanck = np.pad(fisherPlanck,pad_width=((0,numLeft),(0,numLeft)),mode="constant",constant_values=0.)

# pl = Plotter()
# pl.plot2d(cov2corr(fisherPlanck))
# pl.done("output/fisherplanck.png")
# sys.exit()


#f7 = np.loadtxt("output/fisherSigma8S4-7m_CMB_all_wstep.txt")+fisherPlanck #[11:,11:]#+fisherPlanck
#f5 = np.loadtxt("output/fisherSigma8S4-5m_CMB_all_wstep.txt")+fisherPlanck 

f7 = np.loadtxt("fisherSigma8S4-7m_owl2_wstep.txt")+fisherPlanck #[11:,11:]#+fisherPlanck
f5 = np.loadtxt("fisherSigma8S4-5m_owl2_wstep.txt")+fisherPlanck 

# print f7.shape[0]
# print f7.shape[0]-zout.size

# sys.exit()

print f7.shape





i7 = np.linalg.inv(f7)
i5 = np.linalg.inv(f5)

err7 = np.sqrt(np.diagonal(i7))[13:]
err5 = np.sqrt(np.diagonal(i5))[13:]

print err7
#print e7.shape

pl = Plotter(labelX="$z$",labelY="Error on $\sigma_8(z)/\sigma_8(0)$")
pl.addErr(zout+0.02,zout*0.+1.,yerr=err7,label="S4-7m")
pl.addErr(zout,zout*0.+1.,yerr=err5,label="S4-5m")
pl.legendOn()
pl.done("output/s8errsowl.png")



corr = cov2corr(f7)    
pl = Plotter()
pl.plot2d(corr)
pl.done("output/corrowl.png")
