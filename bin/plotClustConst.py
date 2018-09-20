from __future__ import print_function
from __future__ import division
from builtins import zip
from builtins import range
from past.utils import old_div
import numpy as np
import matplotlib.pyplot as plt
from getdist import plots, MCSamples
import glob
import argparse
from scipy.interpolate import interp1d

parser = argparse.ArgumentParser(description='Plot Cluster Chains.')
#parser.add_argument("dir_name", type=str,help='dir of run.')
#parser.add_argument("chain_name", type=str,help='Root name of run.')
#parser.add_argument("-N", "--nruns",     type=int,  default=int(1e6),help="Number of chains.")
#parser.add_argument("-N", "--nruns",     type=int,  default=int(1e6),help="Number of chains.")
parser.add_argument("-t", "--test", action='store_true',help='Do a simtest quickly with 3 params.')
parser.add_argument("-s", "--s8test", action='store_true',help='Do a simtest quickly with s8.')

args = parser.parse_args()

def bring_together_samples(home,chain_name,burnin):

    f = glob.glob(home+chain_name+"*.dat")
    #f = home+chain_name+"_"+str(0)+".dat"
    #all_samps = np.loadtxt(f[0])
    #all_samps = all_samps[burnin:,:]
    for i in range(len(f)):
        a = np.loadtxt(f[i])[burnin:,:]
        if (i == 0): all_samps = a
        all_samps = np.append(all_samps,a,axis=0)

    Om_samps = np.array([old_div((all_samps[:,0] + all_samps[:,1]),(old_div(all_samps[:,2],100.))**2)])
    all_samps = np.concatenate((all_samps,Om_samps.T),axis=1)

    return all_samps

def load_single_sample(home,chain_name,burnin):

    f = home+chain_name
    all_samps = np.loadtxt(f)[burnin:,:]

    return all_samps

outdir = "/Users/nab/Desktop/Projects/ACTPol_Cluster_Like/"
#dir_name = "/Users/nab/Desktop/Projects/ACTPol_Cluster_Like/v9working_run/"
#dir_name1 = "/Users/nab/Desktop/Projects/ACTPol_Cluster_Like/highbias_v10/"
#dir_name2 = "/Users/nab/Desktop/Projects/ACTPol_Cluster_Like/lowbias_v10/"


if args.test:
    dir_name1 = "/Users/nab/Desktop/Projects/ACTPol_Cluster_Like/ACT_chains/"
    chain1 = "sz_chain_test_sim_pars_v1_0.dat"
    burnins = 2000
    
    names = ['As','omch2','ombh2','s8']
    labels =  ['A_s','\Omega_c h2','\Omega_b h^2','\sigma_8']
    out1 = load_single_sample(dir_name1,chain1,burnins)
    samples1 = MCSamples(samples=out1,names = names, labels = labels)

    p = samples1.getParams() 
    samples1.addDerived(old_div((p.omch2 + p.ombh2),0.7**2), name='om', label='\Omega_M')
    
    plt.figure()
    g = plots.getSubplotPlotter()
    g.triangle_plot([samples1], params=['omch2','ombh2','s8','om'], filled=True)
    plt.savefig(outdir+"simtest_parsTestv1.png")
    
elif args.s8test:
    dir_name1 = "/Users/nab/Desktop/Projects/ACTPol_Cluster_Like/ACT_chains/"
    chain1 = "sz_chain_test_chains_v4_0.dat"
    #chain1 = "sz_likelival_test_s8_mock.dat"
    burnins = 0 #,2000

    #likefile = dir_name1 + 'sz_likelival_test_s8_mock_D56Equ_v3.dat'
    likefile = dir_name1 + 'sz_likelival_mockCat_v7.dat'
    #likefile = dir_name1 + 'sz_likelival_test_chains_v4.dat'
    #likefile = dir_name1 + 'sz_likelival_test_s8_mock.dat'
    
    As1D,like1D,  = np.loadtxt(likefile)
    
    names = ['As','s8']
    labels =  ['A_s','\sigma_8']
    out1 = load_single_sample(dir_name1,chain1,burnins)
    samples1 = MCSamples(samples=out1,names = names, labels = labels)
    
    indsort = np.argsort(out1[:,0])
    #print indsort
    
    fint = interp1d(out1[:,0][indsort], out1[:,1][indsort],fill_value='extrapolate')
    #print out1[:,0][indsort]
    #print As1D

    indmin = np.argmax(like1D)
    print(As1D[indmin])

    print(fint(As1D[indmin]))
    
    #print len(indsort)

    print(As1D.shape)

    plt.figure()
    #plt.plot(fint(As1D), like1D)
    plt.plot(As1D, like1D)
    plt.savefig(outdir+"liketest_s8mockv7.png")
    #plt.figure()
    #g = plots.getSubplotPlotter()
    #g.triangle_plot([samples1], params=['As','s8'], filled=True)
    #plt.savefig(outdir+"simtest_s8mockv1.png")
    
    

else:
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

#print out1.shape
    
    samples1 = MCSamples(samples=out1,names = names, labels = labels)
    samples2 = MCSamples(samples=out2,names = names, labels = labels)
    
    #print out1[-1:,:]
     
    print(samples2.getTable(limit=1).tableTex())
    
    plt.figure()
    g = plots.getSubplotPlotter()
    g.triangle_plot([samples1,samples2], params=['massbias','yslope','scat','s8','om'], filled=True)
    plt.savefig(outdir+"v10_and_v11_testv2.png")
    print(samples2.getTable(limit=1).tableTex())

if not args.s8test:

    print(samples1.getTable(limit=1).tableTex())
    
    constraints = [(v[1], v[2]-v[1], v[1]-v[0]) for v in zip(*np.percentile(out1, [16, 50, 84],axis=0))]
    print(constraints)
    
