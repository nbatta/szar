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
#dir_name = "/Users/nab/Desktop/Projects/ACTPol_Cluster_Like/v9working_run/"
dir_name1 = "/Users/nab/Desktop/Projects/ACTPol_Cluster_Like/v10updated/"
dir_name2 = "/Users/nab/Desktop/Projects/ACTPol_Cluster_Like/v11updated/"
#chain = "sz_chain_production_v9"
chain1 = "sz_chain_production_v10"
chain2 = "sz_chain_production_v11"
burnins = 1600

names = ['omch2','ombh2','H0','As','ns','massbias','yslope','scat','s8','om']
labels =  ['\Omega_c h2','\Omega_b h^2','H_0','A_s','n_s','1-b','B','\sigma_{YM}','\sigma_8','\Omega_M']
lims= [[0.04,0.44],[0.019,0.027],]


out1 = bring_together_samples(dir_name1,chain1,burnins)
out2 = bring_together_samples(dir_name2,chain2,burnins)

constraints = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(out1, [16, 50, 84],axis=0)))

print constraints
#print out1.shape

samples1 = MCSamples(samples=out1,names = names, labels = labels)
samples2 = MCSamples(samples=out2,names = names, labels = labels)

print out1[-1:,:]

print(samples1.getTable(limit=1).tableTex())
print(samples2.getTable(limit=1).tableTex())

#plt.figure()
#g = plots.getSubplotPlotter()
#g.triangle_plot([samples1,samples2], params=['massbias','yslope','scat','s8','om'], filled=True)
#plt.savefig(outdir+"v10_and_v11_test.png")


