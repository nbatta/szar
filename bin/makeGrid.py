import matplotlib
matplotlib.use('Agg')
from mpi4py import MPI
import numpy as np
from alhazen.halos import NFWMatchedFilterSN
from szlib.szcounts import ClusterCosmology,SZ_Cluster_Model,Halo_MF

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()    



if rank==0:

    import sys
    from ConfigParser import SafeConfigParser 
    import cPickle as pickle


    import argparse

    parser = argparse.ArgumentParser(description='Make an M,z grid using MPI. Currently implements CMB lensing \
    matched filter and SZ variance.')
    parser.add_argument('expName', type=str,help='The name of the experiment in input/pipeline.ini')
    parser.add_argument('gridName', type=str,help='The name of the grid in input/pipeline.ini')
    parser.add_argument('lensName', nargs='?',type=str,help='The name of the CMB lensing calibration in input/pipeline.ini. Not required if using --skip-lensing option.',default="")
    parser.add_argument('--skip-lensing', dest='skipLens', action='store_const',
                        const=True, default=False,
                        help='Skip CMB lensing matched filter.')

    parser.add_argument('--skip-sz', dest='skipSZ', action='store_const',
                        const=True, default=False,
                        help='Skip SZ variance.')

    args = parser.parse_args()


    expName = args.expName
    gridName = args.gridName
    lensName = args.lensName

    doLens = not(args.skipLens)
    doSZ = not(args.skipSZ)


    if doLens: assert lensName!="", "ERROR: You didn't specify a lensName. If you don't want to do lensing, add --skip-lensing."

    assert doLens or doSZ, "ERROR: Nothing to do."





    iniFile = "input/pipeline.ini"
    Config = SafeConfigParser()
    Config.optionxform=str
    Config.read(iniFile)

    fparams = {}   
    for (key, val) in Config.items('params'):
        if ',' in val:
            param, step = val.split(',')
            fparams[key] = float(param)
        else:
            fparams[key] = float(val)

    from orphics.tools.io import dictFromSection, listFromConfig

    ms = listFromConfig(Config,gridName,'mexprange')
    mgrid = np.arange(ms[0],ms[1],ms[2])
    zs = listFromConfig(Config,gridName,'zrange')
    zgrid = np.arange(zs[0],zs[1],zs[2])

    beam = listFromConfig(Config,expName,'beams')
    noise = listFromConfig(Config,expName,'noises')
    freq = listFromConfig(Config,expName,'freqs')
    lkneeT,lkneeP = listFromConfig(Config,expName,'lknee')
    alphaT,alphaP = listFromConfig(Config,expName,'alpha')
    tellmin,tellmax = listFromConfig(Config,expName,'tellrange')
    pellmin,pellmax = listFromConfig(Config,expName,'pellrange')
    lmax = int(Config.getfloat(expName,'lmax'))

    if doLens:
        pols = Config.get(lensName,'polList').split(',')
        delens = Config.getboolean(lensName,'delens')
        freq_to_use = Config.getfloat(lensName,'freq')
        ind = np.where(np.isclose(freq,freq_to_use))
        beamFind = np.array(beam)[ind]
        noiseFind = np.array(noise)[ind]
        assert beamFind.size==1
        assert noiseFind.size==1
        beamX = beamY = beamFind[0]
        noiseTX = noiseTY = noiseFind[0]


        from orphics.tools.cmb import loadTheorySpectraFromCAMB
        import flipper.liteMap as lm
        from alhazen.quadraticEstimator import NlGenerator,getMax
        deg = 10.
        px = 0.5
        dell = 20
        gradCut = 2000
        kmin = 100
        cambRoot = "data/ell28k_highacc"
        theory = loadTheorySpectraFromCAMB(cambRoot,unlensedEqualsLensed=False,useTotal=False,lpad=9000)
        lmap = lm.makeEmptyCEATemplate(raSizeDeg=deg, decSizeDeg=deg,pixScaleXarcmin=px,pixScaleYarcmin=px)

        Nleach = {}
        kmaxes = []
        for polComb in pols:
            kmax = getMax(polComb,tellmax,pellmax)
            bin_edges = np.arange(kmin,kmax,dell)+dell

            myNls = NlGenerator(lmap,theory,bin_edges,gradCut=gradCut)
            myNls.updateNoise(beamX,noiseTX,np.sqrt(2.)*noiseTX,tellmin,tellmax,pellmin,pellmax,beamY=beamY,noiseTY=noiseTY,noisePY=np.sqrt(2.)*noiseTY,lkneesX=(lkneeT,lkneeP),lkneesY=(lkneeT,lkneeP),alphasX=(alphaT,alphaP),alphasY=(alphaT,alphaP))

            if (polComb=='EB' or polComb=='TB') and (delens):
                ls, Nls, eff = myNls.iterativeDelens(polComb,1.0,True)
            else:
                ls,Nls = myNls.getNl(polComb=polComb,halo=True)

            Nleach[polComb] = (ls,Nls)
            kmaxes.append(kmax)

        bin_edges = np.arange(kmin,max(kmaxes),dell)+dell
        Nlmvinv = 0.
        from scipy.interpolate import interp1d

        from orphics.tools.io import Plotter
        ellkk = np.arange(2,9000,1)
        Clkk = theory.gCl("kk",ellkk)    
        pl = Plotter(scaleY='log',scaleX='log')
        pl.add(ellkk,4.*Clkk/2./np.pi)


        for polComb in pols:
            ls,Nls = Nleach[polComb]
            nlfunc = interp1d(ls,Nls,bounds_error=False,fill_value=np.inf)
            Nleval = nlfunc(bin_edges)
            Nlmvinv += np.nan_to_num(1./Nleval)
            pl.add(ls,4.*Nls/2./np.pi,label=polComb)

        Nlmv = np.nan_to_num(1./Nlmvinv)
        ls = bin_edges[1:-1]
        Nls = Nlmv[1:-1]

        pl.add(ls,4.*Nls/2./np.pi,ls="--")
        pl.legendOn(loc='lower left',labsize=10)
        pl.done("output/Nl_"+expName+lensName+".png")
        #ls,Nls = np.loadtxt("data/LA_pol_Nl.txt",unpack=True,delimiter=",")
        # print ls,Nls
        np.savetxt("data/nlsave_"+expName+"_"+lensName+".txt",np.vstack((ls,Nls)).transpose())
    else:
        ls = None
        Nls = None
    
    constDict = dictFromSection(Config,'constants')
    clttfile = Config.get('general','clttfile')


    if doSZ:
        clusterDict = dictFromSection(Config,'cluster_params')

        cc = ClusterCosmology(fparams,constDict,clTTFixFile=clttfile)
        if doSZ:
            HMF = Halo_MF(cc,mgrid,zgrid)
            kh = HMF.kh
            pk = HMF.pk

    else:
        clusterDict = None
        kh = None
        pk = None
        

else:
    doLens = None
    doSZ = None
    mgrid = None
    zgrid = None
    ls = None
    Nls = None
    fparams = None
    constDict = None
    clttfile = None
    clusterDict = None
    beam = None
    noise = None
    freq = None
    lkneeT = None
    alphaT = None
    kh = None
    pk = None

if rank==0: print "Broadcasting..."
doLens = comm.bcast(doLens, root = 0)
doSZ = comm.bcast(doSZ, root = 0)
mgrid = comm.bcast(mgrid, root = 0)
zgrid = comm.bcast(zgrid, root = 0)
ls = comm.bcast(ls, root = 0)
Nls = comm.bcast(Nls, root = 0)
fparams = comm.bcast(fparams, root = 0)
constDict = comm.bcast(constDict, root = 0)
clttfile = comm.bcast(clttfile, root = 0)
clusterDict = comm.bcast(clusterDict, root = 0)
beam = comm.bcast(beam, root = 0)
noise = comm.bcast(noise, root = 0)
freq = comm.bcast(freq, root = 0)
lkneeT = comm.bcast(lkneeT, root = 0)
alphaT = comm.bcast(alphaT, root = 0)
kh = comm.bcast(kh, root = 0)
pk = comm.bcast(pk, root = 0)
if rank==0: print "Broadcasted."


cc = ClusterCosmology(fparams,constDict,clTTFixFile=clttfile)
if doSZ:
    HMF = Halo_MF(cc,mgrid,zgrid,kh=kh,powerZK=pk)
    SZCluster = SZ_Cluster_Model(cc,clusterDict,rms_noises = noise,fwhms=beam,freqs=freq,lknee=lkneeT,alpha=alphaT)

numms = mgrid.size
numzs = zgrid.size
if doLens: MerrGrid = np.zeros((numms,numzs))
if doSZ: siggrid = np.zeros((numms,numzs))
numes = numms*numzs



indices = np.arange(numes)
splits = np.array_split(indices,numcores)

if rank==0:
    lengths = [len(split) for split in splits]
    mintasks = min(lengths)
    maxtasks = max(lengths)
    print "I have ",numcores, " cores to work with."
    print "And I have ", numes, " tasks to do."
    print "Each worker gets at least ", mintasks, " tasks and at most ", maxtasks, " tasks."
    buestguess = (0.5*int(doSZ)+2.0*int(doLens))*maxtasks
    print "My best guess is that this will take ", buestguess, " seconds."
    print "Starting the slow part..."


mySplitIndex = rank

mySplit = splits[mySplitIndex]

for index in mySplit:
            
    mindex,zindex = np.unravel_index(index,(numms,numzs))
    mass = mgrid[mindex]
    z = zgrid[zindex]

    if doLens:
        kmax = 8000
        overdensity = 500
        critical = True
        atClusterZ = True
        concentration = 1.18
        MerrGrid[mindex,zindex] = 1./NFWMatchedFilterSN(cc,mass,concentration,z,ells=ls,Nls=Nls,kellmax=kmax,overdensity=overdensity,critical=critical,atClusterZ=atClusterZ,saveId=None)
    if doSZ:
        var = SZCluster.quickVar(mass,z)
        siggrid[mindex,zindex] = np.sqrt(var)




if rank!=0:
    if doLens:
        MerrGrid = MerrGrid.astype(np.float64)
        comm.Send(MerrGrid, dest=0, tag=77)
    if doSZ:
        siggrid = siggrid.astype(np.float64)
        comm.Send(siggrid, dest=0, tag=78)
    
else:
    print "Waiting for workers..."

    import cPickle as pickle

    if doLens:
        for i in range(1,numcores):
            data = np.zeros(MerrGrid.shape, dtype=np.float64)
            comm.Recv(data, source=i, tag=77)
            MerrGrid += data

        pickle.dump((mgrid,zgrid,MerrGrid),open("data/lensgrid_"+expName+"_"+gridName+"_"+lensName+".pkl",'wb'))

    if doSZ:
        for i in range(1,numcores):
            data = np.zeros(siggrid.shape, dtype=np.float64)
            comm.Recv(data, source=i, tag=78)
            siggrid += data


        pickle.dump((mgrid,zgrid,siggrid),open("data/szgrid_"+expName+"_"+gridName+".pkl",'wb'))
