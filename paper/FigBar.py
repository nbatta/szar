import orphics.tools.io as io
import numpy as np
import sys, os

out_dir = os.environ['WWW']+"paper/"


objects = ('mnu', 'mnu+w', 'mnu+w+wa')
y_pos = np.arange(len(objects))
performance1 = [34.26,35.84,36.07]
performance2 = [25.3,31.03,31.08]
performance3 = [16.5,23.83,23.89]

performance1o = [27.12,27.28,27.31]
performance2o = [26.65,27.12,27.24]
performance3o = [17.27,17.52,17.73]


performance1j = [26.02,26.51,26.53]
performance2j = [23.94,26.08,26.09]
performance3j = [14.26,16.05,16.05]


pl = io.Plotter(labelY='$\\sigma(m_{\\nu})$')
# pl._ax.bar(y_pos, performance1, align='center', alpha=0.5,label="S4")
# pl._ax.bar(y_pos, performance2, align='center', alpha=0.5,label="S4+DESI")
# pl._ax.bar(y_pos, performance3, align='center', alpha=0.5,label="S4+DESI+bbtau")


pl._ax.bar(y_pos, performance1o, align='center', alpha=0.5,label="S4")
pl._ax.bar(y_pos, performance2o, align='center', alpha=0.5,label="S4+DESI")
pl._ax.bar(y_pos, performance3o, align='center', alpha=0.5,label="S4+DESI+bbtau")

# pl._ax.bar(y_pos, performance1j, align='center', alpha=0.5,label="S4")
# pl._ax.bar(y_pos, performance2j, align='center', alpha=0.5,label="S4+DESI")
# pl._ax.bar(y_pos, performance3j, align='center', alpha=0.5,label="S4+DESI+bbtau")


#pl._fig.xticks(y_pos, objects)
pl._ax.set_xticks(y_pos)#y_pos, objects)
pl._ax.set_xticklabels(objects)
pl._ax.set_ylim(0,60)
pl._ax.axhline(y=20,ls="--",color="k")
pl._ax.axhline(y=12,ls="--",color="k")
pl.legendOn(labsize=12)
pl.done(out_dir+"FigBarB.pdf")
