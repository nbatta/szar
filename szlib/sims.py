import matplotlib
matplotlib.use('Agg')
import ConfigParser
import numpy as np
import matplotlib.pyplot as plt
import glob

class BattagliaSims(object):

    def __init__(self,clusterCosmology,rootPath="/astro/astronfs01/workarea/msyriac/clusterSims/Battaglia/"):

        self.cc = clusterCosmology

        self.root = rootPath
        alist = np.loadtxt(rootPath + 'outputSel.txt')
        self.zlist = np.array((1./alist)-1.)
        self.snaplist = np.arange(55,55-len(self.zlist),-1)
        self.snapToZ = lambda s: self.zlist[self.snaplist==s] 

    def mapReader(self):
        PIX = 2048 # all files are 2048 x 2048
        filelist = glob.glob(self.root+"GEN_Cluster_MassDM*_snap??_comovFINE.d")
        print len(filelist)
        i = 0
        for fileDM in filelist:

            fileStar = fileDM.replace("MassDM_","MassStar_")
            fileGas = fileDM.replace("MassDM_","MassGas_")
            fileSZ = fileDM.replace("MassDM_","")

            zext = fileDM.find("snap")
            snap = int(fileDM[zext+4:zext+6])
            z = self.snapToZ(snap)[0]
            
            mapList = []
            for filen in [fileDM,fileStar,fileGas,fileSZ]:
                with open(filen, 'rb') as fd:
                    temp = np.fromfile(file=fd, dtype=np.float32)

                #reshape array into 2D array
                map = np.reshape(temp,(PIX,PIX))
                mapList.append(map)

            del temp
            yield z,mapList

            #Check map values to make sure they aren't crazy
            #print np.max(map), np.min(map)
            #Check the image
            # zoom = PIX/2 -128
            # plt.imshow((map[zoom:-zoom,zoom:-zoom]),interpolation='nearest')
            # plt.savefig("plots/plot"+str(i)+".png")
            # i+=1



