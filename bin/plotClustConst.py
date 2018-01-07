import numpy as np
import matplotlib.pyplot as plt
from getdist import plots, MCSamples
import glob
#import argparse

#parser = argparse.ArgumentParser(description='Plot Cluster Chains.')
#parser.add_argument("dir_name", type=str,help='dir of run.')
#parser.add_argument("chain_name", type=str,help='Root name of run.')
#parser.add_argument("-N", "--nruns",     type=int,  default=int(1e6),help="Number of chains.")
#args = parser.parse_args()

def bring_together_samples(home,chain_name,burnin):

    f = glob.glob(home+chain_name+"*.dat")
    #f = home+chain_name+"_"+str(0)+".dat"
    #all_samps = np.loadtxt(f[0])
    #all_samps = all_samps[burnin:,:]
    for i in range(len(f)):
        a = np.loadtxt(f[i])[burnin:,:]
        if (i == 0): all_samps = a
        all_samps = np.append(all_samps,a,axis=0)
        
    Om_samps = np.array([(all_samps[:,0] + all_samps[:,1])/(all_samps[:,2]/100.)**2])
    all_samps = np.concatenate((all_samps,Om_samps.T),axis=1)

    return all_samps

outdir = "/Users/nab/Desktop/Projects/ACTPol_Cluster_Like/"
dir_name = "/Users/nab/Desktop/Projects/ACTPol_Cluster_Like/v9working_run/"
chain1 = "sz_chain_production_v9"
burnins = 1600
nruns = 19

names = ['omch2','ombh2','H0','As','ns','massbias','yslope','scat','s8','om']
labels =  ['\omega_c h2','\omega_b h^2','H_0','A_s','n_s','1-b','B','\sigma_{YM}','\sigma_8','\Omega_M']

out1 = bring_together_samples(dir_name,chain1,burnins)

print out1.shape

samples1 = MCSamples(samples=out1,names = names, labels = labels)

plt.figure()
g = plots.getSubplotPlotter()
g.triangle_plot([samples1], filled=True)
plt.savefig(outdir+"v9_test_script.png")


