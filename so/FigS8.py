import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from configparser import SafeConfigParser 
import pickle as pickle
import sys
import numpy as np
from orphics.io import Plotter,dict_from_section,list_from_config
    

s8files = sys.argv[1]

pl = Plotter(xlabel="$z$",ylabel="$\sigma_8(z)/\sigma_8(z)_{w=-1}$",ftsize=16)
from matplotlib.patches import Rectangle

cols = (['C0'],['C1'])
labs = (['Baseline $f_{\\mathrm{sky}}=0.4$'],['Goal $f_{\\mathrm{sky}}=0.4$'])

exps = ['base','goal']
bins = {}
for s8file,colList,lablist,exp in zip(s8files.split(','),cols,labs,exps):

    bins[exp] = []

    def getFisher():

        paramList,FisherTot = pickle.load(open(s8file,'rb'))
        return paramList,FisherTot



    iniFile = "input/pipeline.ini"
    Config = SafeConfigParser()
    Config.optionxform=str
    Config.read(iniFile)
    bigDataDir = Config.get('general','bigDataDirectory')

    fishSection = "lcdm-paper"
    noatm = ""
    cal = "CMB_all"
    derivSet = "0.6"
    gridName = "grid-default"


    CB_color_cycle = ['#1C110A','#E4D6A7','#E9B44C','#9B2915','#50A2A7'][::-1]
    import matplotlib as mpl
    # mpl.rcParams['axes.color_cycle'] = CB_color_cycle


    cosmoFisher = Config.get('fisher-'+fishSection,'saveSuffix')
    origParams = Config.get('fisher-'+fishSection,'paramList').split(',')


    """RES STUDY"""
    ps,cmbfisher0 = getFisher()


    pindex = ps.index("S8Z0")
    print((ps[pindex:]))


    from szar.counts import getA
    fparams = {}   # the 
    for (key, val) in Config.items('params'):
        if ',' in val:
            param, step = val.split(',')
            fparams[key] = float(param)
        else:
            fparams[key] = float(val)

    constDict = dict_from_section(Config,'constants')
    zs = list_from_config(Config,gridName,'zrange')
    z_edges = np.arange(zs[0],zs[1]+zs[2],zs[2])
    zrange = (z_edges[1:]+z_edges[:-1])/2.

    s81,As1 = getA(fparams,constDict,zrange)
    print(s81)
    s81zs = As1*s81
    fparams['w0']=-0.97
    s82,As2 = getA(fparams,constDict,zrange)
    print(s82)
    s82zs = As2*s82


    outDir = "/gpfs01/astro/www/msyriac/paper/"


    zbins = zrange 
    #zbins = np.append(np.arange(2.,2.5,0.5),3.0)
    print(zrange)



    #colList = ['C0']


    #lablist = ['SO']
    currentAxis = plt.gca()



    #zbins = zrange 
    zbins = np.append(np.arange(0.,2.5,0.5),3.0)
    #zbins = np.array([0.,1.,2.,3.0])
    print(zbins)
    #sys.exit()


    #colList = ['C5']

    currentAxis = plt.gca()
    for i,(f,lab,col) in enumerate(zip([cmbfisher0],lablist,colList)):
        inv = np.linalg.inv(f)
        err = np.sqrt(np.diagonal(inv))[pindex:]
        zcents = []
        errcents = []
        xerrs = []
        ms8 = []
        pad = 0.05
        s80mean = np.mean(s81zs[np.logical_and(zrange>=zbins[0],zrange<zbins[1])])
        yerrsq0 = (1./sum([1/x**2. for x in err[np.logical_and(zrange>=zbins[0],zrange<zbins[1])]]))

        for zleft,zright in zip(zbins[:-1],zbins[1:]):
            errselect = err[np.logical_and(zrange>=zleft,zrange<zright)]
            zcent = (zleft+zright)/2.
            zcents.append(zcent)
            yerr = np.sqrt(1./sum([1/x**2. for x in errselect]))
            xerr = (zright-zleft)/2.
            xerrs.append(xerr)
            s8now = np.mean(s81zs[np.logical_and(zrange>=zleft,zrange<zright)])
            print((lab,zleft,zright, yerr,s8now, yerr*100./s8now, "%"))
            #s8now = np.mean(s81zs[np.logical_and(zrange>=zleft,zrange<zright)])/s81
            #yerrsq = (1./sum([1/x**2. for x in errselect]))
            #yerr = (s8now/s80mean)*np.sqrt(yerrsq/s8now**2. + yerrsq0/s80mean**2.)
            errcents.append(yerr)
            ms8.append(s8now)
            currentAxis.add_patch(Rectangle((zcent - xerr+pad, 1. - yerr/s8now), 2*xerr-pad/2., 2.*yerr/s8now, facecolor=col,alpha=1.0))
            bins[exp].append((zcent,xerr,yerr,s8now))
        print("=====================")
        pl._ax.fill_between(zrange, 1., 1.,label=lab,alpha=0.75,color=col)

pickle.dump(bins,open("saved_s8_data.pkl",'wb'))
#pl.add(zrange,s82zs/s81zs,label="$w=-0.97$",color='red',alpha=0.5)
pl.add(zrange,s81zs/s81zs,color='white',alpha=0.5,ls="--")#,label="$w=-1$")

# pl.add(zrange,s82zs/s81zs/s82*s81,label="$w=-0.97$",color='red',alpha=0.5)
# pl.add(zrange,s81zs*0.+1.,label="$w=-1$",color='black',alpha=0.5,ls="--")


pl.legend(labsize=12,loc="lower left")
#pl._ax.set_ylim(0.88,1.12) # res
pl._ax.set_ylim(0.93,1.07) # fsky
#pl._ax.text(0.8,.82,"Madhavacheril et. al. in prep")
pl.done("FigSOS8.pdf")
#pl.done(outDir+"s8SO.png")
