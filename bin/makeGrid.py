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
    version = Config.get('general','version')

    fparams = {}   
    for (key, val) in Config.items('params'):
        if ',' in val:
            param, step = val.split(',')
            fparams[key] = float(param)
        else:
            fparams[key] = float(val)

    from orphics.tools.io import dictFromSection, listFromConfig

    bigDataDir = Config.get('general','bigDataDirectory')

    ms = listFromConfig(Config,gridName,'mexprange')
    Mexp_edges = np.arange(ms[0],ms[1]+ms[2],ms[2])
    zs = listFromConfig(Config,gridName,'zrange')
    z_edges = np.arange(zs[0],zs[1]+zs[2],zs[2])


    M_edges = 10**Mexp_edges
    M = (M_edges[1:]+M_edges[:-1])/2.
    mgrid = np.log10(M)

    zgrid = (z_edges[1:]+z_edges[:-1])/2.

    
    beam = listFromConfig(Config,expName,'beams')
    noise = listFromConfig(Config,expName,'noises')
    freq = listFromConfig(Config,expName,'freqs')
    lkneeT,lkneeP = listFromConfig(Config,expName,'lknee')
    alphaT,alphaP = listFromConfig(Config,expName,'alpha')
    tellmin,tellmax = listFromConfig(Config,expName,'tellrange')
    pellmin,pellmax = listFromConfig(Config,expName,'pellrange')
    lmax = int(Config.getfloat(expName,'lmax'))
    constDict = dictFromSection(Config,'constants')

    if doLens:
        pols = Config.get(lensName,'polList').split(',')
        miscentering = Config.getboolean(lensName,'miscenter')
        delens = Config.getboolean(lensName,'delens')
        freq_to_use = Config.getfloat(lensName,'freq')
        ind = np.where(np.isclose(freq,freq_to_use))
        beamFind = np.array(beam)[ind]
        noiseFind = np.array(noise)[ind]
        assert beamFind.size==1
        assert noiseFind.size==1
        


        from orphics.tools.cmb import loadTheorySpectraFromCAMB
        import flipper.liteMap as lm
        from alhazen.quadraticEstimator import NlGenerator,getMax
        deg = 10.
        px = 0.5
        dell = 20
        gradCut = 2000
        kmin = 100
        
        from orphics.theory.cosmology import Cosmology
        cc = Cosmology(lmax=8000,pickling=True)
        theory = cc.theory

        #cambRoot = "data/ell28k_highacc"
        #theory = loadTheorySpectraFromCAMB(cambRoot,unlensedEqualsLensed=False,useTotal=False,lpad=9000)
        lmap = lm.makeEmptyCEATemplate(raSizeDeg=deg, decSizeDeg=deg,pixScaleXarcmin=px,pixScaleYarcmin=px)

        
        testFreq = freq_to_use
        from szlib.szcounts import fgNoises
        fgs = fgNoises(constDict,ksz_battaglia_test_csv="data/ksz_template_battaglia.csv",tsz_battaglia_template_csv="data/sz_template_battaglia.csv")
        tcmbmuk = constDict['TCMB'] * 1.0e6
        ksz = lambda x: fgs.ksz_temp(x)/x/(x+1.)*2.*np.pi/ tcmbmuk**2.
        radio = lambda x: fgs.rad_ps(x,testFreq,testFreq)/x/(x+1.)*2.*np.pi/ tcmbmuk**2.
        cibp = lambda x: fgs.cib_p(x,testFreq,testFreq) /x/(x+1.)*2.*np.pi/ tcmbmuk**2.
        cibc = lambda x: fgs.cib_c(x,testFreq,testFreq)/x/(x+1.)*2.*np.pi/ tcmbmuk**2.
        tsz = lambda x: fgs.tSZ(x,testFreq,testFreq)/x/(x+1.)*2.*np.pi/ tcmbmuk**2.

        fgFunc = lambda x: ksz(x)+radio(x)+cibp(x)+cibc(x)+tsz(x)
        #fgFunc = None

        
        Nleach = {}
        kmaxes = []
        for polComb in pols:
            X,Y = polComb
            
            if X=='T':
                beamX = 5.0
                noiseTX = 45.0
                noisePX = np.sqrt(2.)*noiseFind[0] # this doesn't matter
                slkneeTX = 0.
                slkneePX = 0. # this doesn't matter
                salphaTX = 1.
                salphaPX = 1. # this doesn't matter
                tellminX = 2.
                tellmaxX = 3000.
                pellminX = 2.
                pellmaxX = 3000.
        
                beamY = beamFind[0]
                noiseTY = noiseFind[0]
                noisePY = np.sqrt(2.)*noiseFind[0]
                slkneeTY = lkneeT
                slkneePY = lkneeP
                salphaTY = alphaT
                salphaPY = alphaP
                tellminY = tellmin
                tellmaxY = tellmax
                pellminY = pellmin
                pellmaxY = pellmax
            else:
                beamX = beamFind[0]
                noiseTX = noiseFind[0]
                noisePX = np.sqrt(2.)*noiseFind[0]
                slkneeTX = lkneeT
                slkneePX = lkneeP
                salphaTX = alphaT
                salphaPX = alphaP
                tellminX = tellmin
                tellmaxX = tellmax
                pellminX = pellmin
                pellmaxX = pellmax
        
                beamY = beamFind[0]
                noiseTY = noiseFind[0]
                noisePY = np.sqrt(2.)*noiseFind[0]
                slkneeTY = lkneeT
                slkneePY = lkneeP
                salphaTY = alphaT
                salphaPY = alphaP
                tellminY = tellmin
                tellmaxY = tellmax
                pellminY = pellmin
                pellmaxY = pellmax

            
            kmax = getMax(polComb,tellmax,pellmax)
            bin_edges = np.arange(kmin,kmax,dell)+dell

            myNls = NlGenerator(lmap,theory,bin_edges,gradCut=gradCut)
            nTX,nPX,nTY,nPY = myNls.updateNoise(beamX,noiseTX,noisePX,tellminX,tellmaxX,pellminX,pellmaxX,beamY=beamY,noiseTY=noiseTY,noisePY=noisePY,lkneesX=(slkneeTX,slkneePX),lkneesY=(slkneeTY,slkneePY),alphasX=(salphaTX,salphaPX),alphasY=(salphaTY,salphaPY),fgFuncY=fgFunc,tellminY=tellminY,tellmaxY=tellmaxY,pellminY=pellminY,pellmaxY=pellmaxY)

            from orphics.tools.io import Plotter
            from orphics.tools.stats import bin2D
            bin_edges = np.arange(100,8000,10)
            binner = bin2D(myNls.N.modLMap, bin_edges)
            pls, binnedTX = binner.bin(nTX)
            pls, binnedTY = binner.bin(nTY)
            pls, binnedPX = binner.bin(nPX)
            pls, binnedPY = binner.bin(nPY)
            
            pl = Plotter(scaleY='log')
            pl.add(pls,theory.uCl('TT',pls)*pls**2.,alpha=0.3,ls="--")
            pl.add(pls,theory.lCl('TT',pls)*pls**2.)
            pl.add(pls,binnedTX*pls**2.,alpha=0.3)
            pl.add(pls,binnedTY*pls**2.)
            pl.done("output/NlTT_"+expName+lensName+polComb+".png")

            pl = Plotter(scaleY='log')
            pl.add(pls,theory.uCl('EE',pls)*pls**2.,alpha=0.3,ls="--")
            pl.add(pls,theory.lCl('EE',pls)*pls**2.)
            pl.add(pls,binnedPX*pls**2.,alpha=0.3)
            pl.add(pls,binnedPY*pls**2.)
            pl.done("output/NlEE_"+expName+lensName+polComb+".png")


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

        from orphics.tools.stats import bin1D
        binner1d = bin1D(bin_edges)
        ellcls , clkk_binned = binner1d.binned(ellkk,Clkk)

        

        pl.add(ellcls,4.*clkk_binned/2./np.pi,ls="none",marker="x")
        pl.add(ellcls,4.*clkk_binned/2./np.pi,ls="none",marker="x")
        pl.add(ls,4.*Nls/2./np.pi,ls="--")
        np.savetxt(bigDataDir+"nlsave_"+expName+"_"+lensName+".txt",np.vstack((ls,Nls)).transpose())

        Nls += clkk_binned[:-1]
        np.savetxt(bigDataDir+"nlsaveTot_"+expName+"_"+lensName+".txt",np.vstack((ls,Nls)).transpose())
        pl.add(ls,4.*Nls/2./np.pi,ls="-.")
        
        pl.legendOn(loc='lower left',labsize=10)
        pl.done("output/Nl_"+expName+lensName+".png")
        #ls,Nls = np.loadtxt("data/LA_pol_Nl.txt",unpack=True,delimiter=",")
        # print ls,Nls
    else:
        ls = None
        Nls = None
        beamY = None
    
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
    beamY = None
    miscentering = None
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
    Mexp_edges = None
    z_edges = None

if rank==0: print "Broadcasting..."
doLens = comm.bcast(doLens, root = 0)
doSZ = comm.bcast(doSZ, root = 0)
beamY = comm.bcast(beamY, root = 0)
miscentering = comm.bcast(miscentering, root = 0)
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
Mexp_edges = comm.bcast(Mexp_edges, root = 0)
z_edges = comm.bcast(z_edges, root = 0)
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
        if miscentering:
            ray = beamY/2.
        else:
            ray = None
            
        snRet,k500,std = NFWMatchedFilterSN(cc,mass,concentration,z,ells=ls,Nls=Nls,kellmax=kmax,overdensity=overdensity,critical=critical,atClusterZ=atClusterZ,saveId=None,rayleighSigmaArcmin=ray)
        MerrGrid[mindex,zindex] = 1./snRet
    if doSZ:
        var = SZCluster.quickVar(10**mass,z)
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
            print "Waiting for lens ", i ," / ", numcores
            data = np.zeros(MerrGrid.shape, dtype=np.float64)
            comm.Recv(data, source=i, tag=77)
            MerrGrid += data

        pickle.dump((Mexp_edges,z_edges,MerrGrid),open(bigDataDir+"lensgrid_"+expName+"_"+gridName+"_"+lensName+ "_v" + version+".pkl",'wb'))

    if doSZ:
        for i in range(1,numcores):
            print "Waiting for sz ", i," / ", numcores
            data = np.zeros(siggrid.shape, dtype=np.float64)
            comm.Recv(data, source=i, tag=78)
            siggrid += data


        pickle.dump((Mexp_edges,z_edges,siggrid),open(bigDataDir+"szgrid_"+expName+"_"+gridName+ "_v" + version+".pkl",'wb'))
