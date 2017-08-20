import orphics.tools.io as io
import numpy as np
import sys, os
from ConfigParser import SafeConfigParser 
import szar.fisher as sfisher

iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)

out_dir = os.environ['WWW']+"paper/"

expName = "S4-1.0-paper"

calList = ["CMB_all","owl2","CMB_all_joint"]
gridList = ["grid-default","grid-owl2","grid-default"]
fishList = ["mnu-paper","mnu-w0-paper","mnu-w0-wa-paper"]
al = 1.0


objects = ('$\\Sigma m_{\\nu}}$', '$\\Sigma m_{\\nu}$+$w$', '$\\Sigma m_{\\nu}$+$w$+$w_a$')
y_pos = np.arange(len(objects))

fishall = ["","-DESI","-DESI-clkk","-DESI-bbtau","-DESI-cvltau"]
laball = ["S4","S4+DESI","S4+DESI+Clkk","S4+DESI+$\\sigma(\\tau)=0.006$","S4+DESI+$\\sigma(\\tau)=0.002$"]

performance = {}
for cal, grid in zip(calList,gridList):
    performance[cal] = {}
    for fish_suff in fishall:
        performance[cal][fish_suff] = []
        for fish in fishList:
            if "bbtau" in fish_suff:
                tau = 0.006
                fish_suff_pass = "-DESI"
            elif "cvltau" in fish_suff:
                tau=0.002
                fish_suff_pass = "-DESI"
            else:
                tau = None
                fish_suff_pass = fish_suff

            if "clkk" in fish_suff:
                fish_suff_pass = "-DESI"
                do_clkk_ov = True
            else:
                do_clkk_ov = None
            with io.nostdout():
                Fisher, paramList = sfisher.cluster_fisher_from_config(Config,expName,grid,cal,fish+fish_suff_pass,tauOverride=tau,do_clkk_override=do_clkk_ov)
            if fish=="mnu-paper": print paramList
            margerrs = sfisher.marginalized_errs(Fisher,paramList)
            print margerrs['mnu']*1000.
            performance[cal][fish_suff].append(margerrs['mnu']*1000.)
            print margerrs['b_wl']*100.

    pl = io.Plotter(labelY='$\\sigma(\\Sigma m_{\\nu})$ (meV)',ftsize=18)
    for fish_suff,lab in zip(fishall,laball):
        pl._ax.bar(y_pos, performance[cal][fish_suff], align='center', alpha=al,label=lab)
        
    pl._ax.set_xticks(y_pos)#y_pos, objects)
    pl._ax.set_xticklabels(objects)
    pl._ax.set_ylim(0,60)
    pl._ax.axhline(y=20,ls="--",color="k")
    pl._ax.axhline(y=12,ls="--",color="k")
    pl.legendOn(labsize=16)
    pl.done(out_dir+"FigBar"+cal+".pdf")

            
# sys.exit()        

# performance1 = [34.26,35.84,36.07]
# performance2 = [25.3,31.03,31.08]
# performance3 = [16.5,23.83,23.89]

# performance1o = [27.12,27.28,27.31]
# performance2o = [26.65,27.12,27.24]
# performance3o = [17.27,17.52,17.73]


# performance1j = [26.02,26.51,26.53]
# performance2j = [23.94,26.08,26.09]
# performance3j = [14.26,16.05,16.05]

#for cal, grid in zip(calList,gridList):

        
# pl._ax.bar(y_pos, performance['CMB_all'], align='center', alpha=0.5,label="S4")
# pl._ax.bar(y_pos, performance['CMB_all'], align='center', alpha=0.5,label="S4+DESI")
# pl._ax.bar(y_pos, performance['CMB_all'], align='center', alpha=0.5,label="S4+DESI+bbtau")


# pl._ax.bar(y_pos, performance1o, align='center', alpha=al,label="S4",color="C0")
# pl._ax.bar(y_pos, performance2o, align='center', alpha=al,label="S4+DESI",color="C1")
# pl._ax.bar(y_pos, performance3o, align='center', alpha=al,label="S4+DESI+bbtau",color="C3")

# pl._ax.bar(y_pos, performance1j, align='center', alpha=0.5,label="S4")
# pl._ax.bar(y_pos, performance2j, align='center', alpha=0.5,label="S4+DESI")
# pl._ax.bar(y_pos, performance3j, align='center', alpha=0.5,label="S4+DESI+bbtau")


#pl._fig.xticks(y_pos, objects)
# pl._ax.set_xticks(y_pos)#y_pos, objects)
# pl._ax.set_xticklabels(objects)
# pl._ax.set_ylim(0,60)
# pl._ax.axhline(y=20,ls="--",color="k")
# pl._ax.axhline(y=12,ls="--",color="k")
# pl.legendOn(labsize=16)
# pl.done(out_dir+"FigBarB.pdf")
