from __future__ import print_function
from __future__ import division
from builtins import str
from builtins import range
from past.utils import old_div
import itertools
from szar.counts import rebinN
import numpy as np
from orphics.io import dict_from_section, list_from_config
from szar.counts import ClusterCosmology,Halo_MF
from szar.szproperties import SZ_Cluster_Model
import pickle as pickle
import traceback


def marginalized_errs(Fisher,paramList):

    # ind = paramList.index("b_wl")
    # print Fisher[ind,ind]
    # Fisher[ind,ind] = 1./0.01**2.
    #print Fisher[ind,ind]
    
    Finv = np.linalg.inv(Fisher)

    errs = np.sqrt(np.diagonal(Finv))
    errDict = {}
    for i,param in enumerate(paramList):
        errDict[param] = errs[i]

    return errDict



def mass_grid_name_cmb_up(bigDataDir,expName,gridName,calName,version):
    return bigDataDir+"lensgridRayUp_"+expName+"_"+gridName+"_"+calName+ "_v" + version+".pkl"

def mass_grid_name_cmb_dn(bigDataDir,expName,gridName,calName,version):
    return bigDataDir+"lensgridRayDn_"+expName+"_"+gridName+"_"+calName+ "_v" + version+".pkl"

def mass_grid_name_cmb(bigDataDir,expName,gridName,calName,version):
    return bigDataDir+"lensgrid_"+expName+"_"+gridName+"_"+calName+ "_v" + version+".pkl"

def mass_grid_name_owl(bigDataDir,calName):
    return bigDataDir+"lensgrid_grid-"+calName+"_"+calName+".pkl"

def hash_func(*argv):
    import hashlib
    hinval = ""
    for arg in argv:
        if isinstance(arg,list):
            hinval += "".join(arg)
        elif isinstance(arg,str):
            hinval += arg
        elif arg is None:
            hinval += "None"
        elif isinstance(arg,bool):
            hinval += str(arg)
        else:
            raise ValueError
            
    return hashlib.md5(hinval).hexdigest()

def pad_fisher(fisher,num_pad):
    return np.pad(fisher,pad_width=((0,num_pad),(0,num_pad)),mode="constant",constant_values=0.)


def get_sovernsquare(expName,gridName,version,qbins):
    sId = expName + "_" + gridName  + "_v" + version
    sovernsquareEach = np.loadtxt(bigDataDir+"sampleVarGrid_"+sId+".txt")
    sovernsquare =  np.dstack([sovernsquareEach]*len(qbins))
    return sovernsquare

def save_id(expName,gridName,calName,version):
    saveId = expName + "_" + gridName + "_" + calName + "_v" + version
    return saveId
def deriv_root(bigDataDir,saveId):
    return bigDataDir+"dNdp_mzq_"+saveId+"_"
def fid_file(bigDataDir,saveId):
    return bigDataDir+"N_mzq_"+saveId+"_fid"+".npy"

def counts_from_config(Config,bigDataDir,version,expName,gridName,mexp_edges,z_edges,lkneeTOverride=None,alphaTOverride=None):
    suffix = ""
    if lkneeTOverride is not None:
        suffix += "_"+str(lkneeTOverride)
    if alphaTOverride is not None:
        suffix += "_"+str(alphaTOverride)
    #mgrid,zgrid,siggrid = pickle.load(open(bigDataDir+"szgrid_"+expName+"_"+gridName+ "_v" + version+suffix+".pkl",'rb'))
    mgrid,zgrid,siggrid = pickle.load(open(bigDataDir+"szgrid_"+expName+"_"+gridName+ "_v" + version+suffix+".pkl",'rb'),encoding='latin1')
    experimentName = expName
    cosmoDict = dict_from_section(Config,"params")
    constDict = dict_from_section(Config,'constants')
    clusterDict = dict_from_section(Config,'cluster_params')
    clttfile = Config.get("general","clttfile")
    cc = ClusterCosmology(cosmoDict,constDict,clTTFixFile = clttfile)

    beam = list_from_config(Config,experimentName,'beams')
    noise = list_from_config(Config,experimentName,'noises')
    freq = list_from_config(Config,experimentName,'freqs')
    lmax = int(Config.getfloat(experimentName,'lmax'))
    lknee = float(Config.get(experimentName,'lknee').split(',')[0])
    alpha = float(Config.get(experimentName,'alpha').split(',')[0])
    fsky = Config.getfloat(experimentName,'fsky')
    SZProf = SZ_Cluster_Model(cc,clusterDict,rms_noises = noise,fwhms=beam,freqs=freq,lknee=lknee,alpha=alpha)

    hmf = Halo_MF(cc,mexp_edges,z_edges)

    hmf.sigN = siggrid.copy()
    Ns = np.multiply(hmf.N_of_z_SZ(fsky,SZProf),np.diff(z_edges).reshape(1,z_edges.size-1))
    return Ns.ravel().sum()


def sel_counts_from_config(Config,bigDataDir,version,expName,gridName,calName,mexp_edges,z_edges,lkneeTOverride=None,alphaTOverride=None,zmin=-np.inf,zmax=np.inf,mmin=-np.inf,mmax=np.inf,recalculate=False,override_params=None):
    suffix = ""
    if lkneeTOverride is not None:
        suffix += "_"+str(lkneeTOverride)
    if alphaTOverride is not None:
        suffix += "_"+str(alphaTOverride)
    mgrid,zgrid,siggrid = pickle.load(open(bigDataDir+"szgrid_"+expName+"_"+gridName+ "_v" + version+suffix+".pkl",'rb'),encoding='latin1')
    experimentName = expName
    cosmoDict = dict_from_section(Config,"params")
    constDict = dict_from_section(Config,'constants')
    clusterDict = dict_from_section(Config,'cluster_params')
    clttfile = Config.get("general","clttfile")
    if override_params is not None:
        for key in override_params.keys():
            cosmoDict[key] = override_params[key]
    # print(cosmoDict)
    cc = ClusterCosmology(cosmoDict,constDict,clTTFixFile = clttfile)

    beam = list_from_config(Config,experimentName,'beams')
    noise = list_from_config(Config,experimentName,'noises')
    freq = list_from_config(Config,experimentName,'freqs')
    lmax = int(Config.getfloat(experimentName,'lmax'))
    lknee = float(Config.get(experimentName,'lknee').split(',')[0])
    alpha = float(Config.get(experimentName,'alpha').split(',')[0])
    fsky = Config.getfloat(experimentName,'fsky')
    SZProf = SZ_Cluster_Model(cc,clusterDict,rms_noises = noise,fwhms=beam,freqs=freq,lknee=lknee,alpha=alpha)

    hmf = Halo_MF(cc,mexp_edges,z_edges)

    hmf.sigN = siggrid.copy()

    saveId = save_id(expName,gridName,calName,version)
    # Fiducial number counts

    if recalculate:
        from . import counts
        # get s/n q-bins
        qs = list_from_config(Config,'general','qbins')
        qspacing = Config.get('general','qbins_spacing')
        if qspacing=="log":
            qbin_edges = np.logspace(np.log10(qs[0]),np.log10(qs[1]),int(qs[2])+1)
        elif qspacing=="linear":
            qbin_edges = np.linspace(qs[0],qs[1],int(qs[2])+1)
        else:
            raise ValueError
        calFile = mass_grid_name_owl(bigDataDir,calName)        
        mexp_edges, z_edges, lndM = pickle.load(open(calFile,"rb"))
        dN_dmqz = hmf.N_of_mqz_SZ(lndM,qbin_edges,SZProf)
        nmzq = counts.getNmzq(dN_dmqz,mexp_edges,z_edges,qbin_edges)
    else:
        nmzq = np.load(fid_file(bigDataDir,saveId))
    nmzq = nmzq*fsky

    zs = (z_edges[1:]+z_edges[:-1])/2.
    zsel = np.logical_and(zs>zmin,zs<=zmax)

    M_edges = 10**mexp_edges
    M = (M_edges[1:]+M_edges[:-1])/2.
    Mexp = np.log10(M)
    msel = np.logical_and(Mexp>mmin,Mexp<=mmax)
    
    Ns = nmzq.sum(axis=-1)[msel,:][:,zsel]
    return Ns #.ravel().sum()


def priors_from_config(Config,expName,calName,fishName,paramList,tauOverride=None):
    fishSection = 'fisher-'+fishName

    try:
        priorNameList = Config.get(fishSection,'prior_names').split(',')
        priorValueList = list_from_config(Config,fishSection,'prior_values')
    except:
        priorNameList = []
        priorValueList = []


    if tauOverride is not None:
        try:
            tauind = priorNameList.index('tau')
            priorValueList[tauind] = tauOverride
        except ValueError:
            priorNameList.append("tau")
            priorValueList.append(tauOverride)

    # if "CMB" in calName:
    #     assert "sigR" not in paramList
    #     paramList.append("sigR")
    #     try:
    #         priorNameList.append("sigR")
    #         beam = list_from_config(Config,expName,'beams')
    #         freq = list_from_config(Config,expName,'freqs')
    #         freq_to_use = Config.getfloat(calName,'freq')
    #         ind = np.where(np.isclose(freq,freq_to_use))
    #         beamFind = np.array(beam)[ind]
    #         priorValueList.append(old_div(beamFind,2.))
    #         print("Added sigR prior ", priorValueList[-1])
    #     except:
    #         traceback.print_exc()
    #         print("Couldn't add sigR prior. Is this CMB lensing? Exiting.")
    #         sys.exit(1)


    # if not("b_wl" in paramList):
    #     print "OWL but b_wl not found in paramList. Adding with a 1% prior."
    #     paramList.append("b_wl")
        #priorNameList.append("b_wl")
        #priorValueList.append(0.01)
    
    # if "owl" in calName:
    #     if not("b_wl" in paramList):
    #         print "OWL but b_wl not found in paramList. Adding with a 1% prior."
    #         paramList.append("b_wl")
    #         priorNameList.append("b_wl")
    #         priorValueList.append(0.01)

    print (paramList, priorNameList, priorValueList)
    return paramList, priorNameList, priorValueList
    

def cluster_fisher_from_config(Config,expName,gridName,calName,fishName,
                               overridePlanck=None,overrideBAO=None,overrideOther=None,pickling=True,s8=False,
                               tauOverride=None,do_clkk_override=None):


    """
    Returns
    1. Fisher - the Fisher matrix
    2. paramList - final parameter list defining the Fisher contents. Might add extra params (e.g. sigR, bwl)

    Accepts
    1. Config - a ConfigParser object containing the ini file contents
    2. expName - name of experiment section in ini file
    3. gridName - name of M,q,z grid definition in ini file
    4. calName - name of weak lensing calibration section
    5. fishName - looks for a section in ini file named "fisher-"+fishName for Fisher options
    6. overridePlanck - Fisher matrix to add to upper left corner of original in place of Planck fisher 
                        matrix specified in ini. Can be zero.
    7. overrideBAO - Fisher matrix to add to upper left corner of original in place of BAO fisher 
                     matrix specified in ini. Can be zero.
    8. overrideOther - Fisher matrix to add to upper left corner of original in place of "other" fisher 
                        matrix specified in ini. Can be zero.
    
    """

    pickling = False #!!!!!
    
    bigDataDir = Config.get('general','bigDataDirectory')
    version = Config.get('general','version') 
    pzcutoff = Config.getfloat('general','photoZCutOff')
    fsky = Config.getfloat(expName,'fsky')
    # Fisher params
    fishSection = 'fisher-'+fishName
    paramList = Config.get(fishSection,'paramList').split(',')

    
    
    zs = list_from_config(Config,gridName,'zrange')
    z_edges = np.arange(zs[0],zs[1]+zs[2],zs[2])


    
    

    saveId = save_id(expName,gridName,calName,version)
    derivRoot = deriv_root(bigDataDir,saveId)
    # Fiducial number counts
    new_z_edges, N_fid = rebinN(np.load(fid_file(bigDataDir,saveId)),pzcutoff,z_edges)#,mass_bin=None)
    N_fid = N_fid*fsky


    # get mass and z grids
    # ms = list_from_config(Config,gridName,'mexprange')
    # mexp_edges = np.arange(ms[0],ms[1]+ms[2],ms[2])
    # M_edges = 10**mexp_edges
    # Masses = (M_edges[1:]+M_edges[:-1])/2.
    # print Masses.shape
    # print N_fid.shape
    # print N_fid.sum()
    # print N_fid[Masses>2.e14,:,:].sum()
    # sys.exit()


    
    print("Effective number of clusters: ", N_fid.sum())

    paramList, priorNameList, priorValueList = priors_from_config(Config,expName,calName,fishName,paramList,tauOverride)

    if s8:
        zrange = old_div((z_edges[1:]+z_edges[:-1]),2.)
        zlist = ["S8Z"+str(i) for i in range(len(zrange))]
        paramList = paramList+zlist

    
    Fisher = getFisher(N_fid,paramList,priorNameList,priorValueList,derivRoot,pzcutoff,z_edges,fsky)

    # Number of non-SZ params (params that will be in Planck/BAO)
    numCosmo = Config.getint(fishSection,'numCosmo')
    numLeft = len(paramList) - numCosmo

    print("param numbers",numCosmo,numLeft)

    try:
        do_cmb_fisher = Config.getboolean(fishSection,"do_cmb_fisher")
    except:
        do_cmb_fisher = False

    try:
        do_clkk_fisher = Config.getboolean(fishSection,"do_clkk_fisher")
    except:
        do_clkk_fisher = False

    if do_clkk_override is not None:
        do_clkk_fisher = do_clkk_override
        
    if do_clkk_fisher:
        assert do_cmb_fisher, "Sorry, currently Clkk fisher requires CMB fisher to be True as well."
        lensName = Config.get(fishSection,"clkk_section")
    else:
        lensName = None
        
    if do_cmb_fisher:

        
        import pyfisher.clFisher as pyfish
        # Load fiducials and derivatives
        cmbDerivRoot = Config.get("general","cmbDerivRoot")
        cmbParamList = paramList[:numCosmo]


        cmb_fisher_loaded = False
        if pickling:
            import time
            hashval = hash_func(cmbParamList,expName,lensName,do_clkk_fisher,time.strftime('%Y%m%d'))
            pkl_file = "output/pickledFisher_"+hashval+".pkl"

            try:
                cmb_fisher = pickle.load(open(pkl_file,'rb'))
                cmb_fisher_loaded = True
                print("Loaded pickled CMB fisher.")
            except:
                pass
            
        if not(cmb_fisher_loaded):
            fidCls = np.loadtxt(cmbDerivRoot+'_fCls.csv',delimiter=',')
            dCls = {}
            for paramName in cmbParamList:
                dCls[paramName] = np.loadtxt(cmbDerivRoot+'_dCls_'+paramName+'.csv',delimiter=',')

            print("Calculating CMB fisher matrix...")
            print(cmbParamList,expName,lensName)
            cmb_fisher = pyfish.fisher_from_config(fidCls,dCls,cmbParamList,Config,expName,lensName)
            if pickling:
                print("Pickling CMB fisher...")
                pickle.dump(cmb_fisher,open(pkl_file,'wb'))

        numLeft = len(paramList) - cmb_fisher.shape[0]
        print("numLeft , " ,numLeft)
        cmb_fisher = pad_fisher(cmb_fisher,numLeft)
    else:
        cmb_fisher = 0.    

    print(len(paramList))
    print(Fisher.shape)

    from orphics import stats
    tszFish = stats.FisherMatrix(Fisher+cmb_fisher,param_list=paramList)
    stats.write_fisher("tsz_fish.txt",tszFish)
    #np.savetxt("tsz.txt",Fisher)

    try:
        otherFishers = Config.get(fishSection,'otherFishers').split(',')
    except:
        traceback.print_exc()
        print("No other fishers found.")
        otherFishers = []

    try:
        external_param_list = Config.get(fishSection,'external_param_list').split(',')
    except:
        traceback.print_exc()
        #external_param_list = "H0,ombh2,omch2,tau,As,ns,mnu,w0,wa".split(',')
        external_param_list = "H0,ombh2,omch2,tau,As,ns".split(',')
        print("No external param list found in fisher section. Assuming ", external_param_list)

    nex = len(external_param_list)
    all_others = np.zeros((nex,nex))
    for otherFisherFile in otherFishers:
        do_other = True
        try:
            other_fisher = np.loadtxt(otherFisherFile)
        except:
            try:
                other_fisher = np.loadtxt(otherFisherFile,delimiter=',') #[:numCosmo,:numCosmo]
            except:
                print("WARNING: Skipped ",otherFisherFile, " either because it was not found or doesn't have enough elements. This means no external fishers have been added!!!")
                do_other = False
                pass
        if do_other:
            #numLeft = len(paramList) - other_fisher.shape[0]
            #other_fisher = pad_fisher(other_fisher,numLeft)
            print("shapes",other_fisher.shape)
            all_others += other_fisher

    otherFish = stats.FisherMatrix(all_others,param_list=external_param_list)
    for p in otherFish.params:
        if p not in paramList:
            otherFish.delete(p)
            print("Deleted ", p , " from external fishers.")
    print("External params : ",otherFish.params)

    #print(otherFish.sigmas()['mnu']*1000.)
    
    retfish = (tszFish + otherFish).ix[paramList,paramList]
    #print(stats.FisherMatrix(retfish.as_matrix(),paramList).marge_var_2param('mnu','w0'))
    return retfish.as_matrix(), paramList



def getFisher(N_fid,paramList,priorNameList,priorValueList,derivRoot,pzcutoff,z_edges,fsky):
    numParams = len(paramList)
    Fisher = np.zeros((numParams,numParams))
    paramCombs = itertools.combinations_with_replacement(paramList,2)
    for param1,param2 in paramCombs:
        i = paramList.index(param1)
        j = paramList.index(param2)
        if not(param1=='tau' or param2=='tau'):
            ppfstr1 = ""
            ppfstr2 = ""
            
            new_z_edges, dN1 = rebinN(np.load(derivRoot+param1+ppfstr1+".npy"),pzcutoff,z_edges)#,mass_bin=None)
            new_z_edges, dN2 = rebinN(np.load(derivRoot+param2+ppfstr2+".npy"),pzcutoff,z_edges)#,mass_bin=None)
            dN1 = dN1*fsky
            dN2 = dN2*fsky


            assert not(np.any(np.isnan(dN1)))
            assert not(np.any(np.isnan(dN2)))
            assert not(np.any(np.isnan(N_fid)))


            with np.errstate(divide='ignore'):
                FellBlock = dN1*dN2*np.nan_to_num(old_div(1.,(N_fid)))#+(N_fid*N_fid*sovernsquare)))
            #Ncollapsed = N_fid.sum(axis=0).sum(axis=-1)
            #print N_fid[np.where(Ncollapsed<1.)].sum() ," clusters fall in bins where N<1"
            #FellBlock[np.where(Ncollapsed<1.)] = 0.
            Fell = FellBlock.sum()
        else:
            Fell = 0.

        if i==j and (param1 in priorNameList):
            priorIndex = priorNameList.index(param1)
            priorVal = old_div(1.,priorValueList[priorIndex]**2.)
        else:
            priorVal = 0.

        Fisher[i,j] = Fell+priorVal
        if j!=i: Fisher[j,i] = Fell

    
    return Fisher
