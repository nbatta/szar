from configparser import SafeConfigParser
from orphics.io import dictFromSection, listFromConfig


iniFile = "input/pipeline.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)



expList = ["dummy-S3","S4-3.0-0.4","S4-2.5-0.4","S4-2.0-0.4","S4-1.5-0.4","S4-1.0-0.4"]

beams={}
noises={}

for i,expName in enumerate(expList):

    
    beams[expName] = listFromConfig(Config,expName,'beams')

    freqs = listFromConfig(Config,expName,'freqs')
    if i==0: freqOrig = list(freqs)
    assert [f==forig for f,forig in zip(freqs,freqOrig)]


latex = ""

print("BEAMS")
print("============")

for i,freq in enumerate(freqs):
    latex += '{:.0f}'.format(freq) 
    for exp in expList:

        val = beams[exp][i]
        if val<1.e-5:
            val = "-"
        else:
            val = '{:.1f}'.format(val)
        latex += " & "+val


    latex += " \\\\ \n"        


print(latex)    


expList = ["dummy-S3","S4-1.5-0.7","S4-1.5-0.4","S4-1.5-0.2","S4-1.5-0.1","S4-1.5-0.05"]

for i,expName in enumerate(expList):

    
    noises[expName] = listFromConfig(Config,expName,'noises')

    freqs = listFromConfig(Config,expName,'freqs')
    if i==0: freqOrig = list(freqs)
    assert [f==forig for f,forig in zip(freqs,freqOrig)]


latex = ""

print("NOISES")
print("============")

for i,freq in enumerate(freqs):
    latex += '{:.0f}'.format(freq) 
    for exp in expList:

        val = noises[exp][i]
        if val<1.e-5:
            val = "-"
        else:
            val = '{:.1f}'.format(val)
        latex += " & "+val


    latex += " \\\\ \n"        


print(latex)    
