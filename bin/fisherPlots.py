from orphics.tools.io import FisherPlots






paramList = Config.get('fisher','paramList').split(',')
paramLatexList = Config.get('fisher','paramLatexList').split(',')
fparams = {} 
for (key, val) in Config.items('params'):
    param = val.split(',')[0]
    fparams[key] = float(param)


fplots = FisherPlots(paramList,paramLatexList,fparams)
fplots.addFisher('planckS4CMBClusters',Fisher)
#fplots.trianglePlot(['H0','mnu','w'],['planckS4OWLClusters','planckS4CMBClusters'],['-','--'])
fplots.plotPair(['mnu','w0'],['planckS4CMBClusters'])
