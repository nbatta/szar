import itertools
from szar.counts import rebinN
import numpy as np


def fisher_from_szar_config(Config,expName,fishName,TCMB=2.7255e6,beamsOverride=None,noisesOverride=None,lkneeTOverride=None,lkneePOverride=None,alphaTOverride=None,alphaPOverride=None,tellminOverride=None,pellminOverride=None,tellmaxOverride=None,pellmaxOverride=None):
    pass



def getFisher(N_fid,paramList,priorNameList,priorValueList,bigDataDir,saveId,pzcutoff,z_edges,fsky):
    numParams = len(paramList)
    Fisher = np.zeros((numParams,numParams))
    paramCombs = itertools.combinations_with_replacement(paramList,2)
    for param1,param2 in paramCombs:
        i = paramList.index(param1)
        j = paramList.index(param2)
        if not(param1=='tau' or param2=='tau'): 
            new_z_edges, dN1 = rebinN(np.load(bigDataDir+"dNdp_mzq_"+saveId+"_"+param1+".npy"),pzcutoff,z_edges)
            new_z_edges, dN2 = rebinN(np.load(bigDataDir+"dNdp_mzq_"+saveId+"_"+param2+".npy"),pzcutoff,z_edges)
            dN1 = dN1[:,:,:]*fsky
            dN2 = dN2[:,:,:]*fsky


            assert not(np.any(np.isnan(dN1)))
            assert not(np.any(np.isnan(dN2)))
            assert not(np.any(np.isnan(N_fid)))


            with np.errstate(divide='ignore'):
                FellBlock = dN1*dN2*np.nan_to_num(1./(N_fid))#+(N_fid*N_fid*sovernsquare)))
            #Ncollapsed = N_fid.sum(axis=0).sum(axis=-1)
            #print N_fid[np.where(Ncollapsed<1.)].sum() ," clusters fall in bins where N<1"
            #FellBlock[np.where(Ncollapsed<1.)] = 0.
            Fell = FellBlock.sum()
        else:
            Fell = 0.

        if i==j and (param1 in priorNameList):
            priorIndex = priorNameList.index(param1)
            priorVal = 1./priorValueList[priorIndex]**2.
        else:
            priorVal = 0.

        Fisher[i,j] = Fell+priorVal
        if j!=i: Fisher[j,i] = Fell+priorVal

    
    return Fisher
