# Appendix C: Collection and Entropy & Response Time Computation

import math
import time
from pox.core import core

log = core.getLogger()


class Entropy(object):
    count = 0
    flag = 0
    destFrequency = {}
    destIP = []
    destEntropy = []
    value = 1
    startList = []
    destIP2 = []
    start = False
    isOK = False
    count_attack = 0
    set_time = []
    countPacket = 0

    def collectStats(self, element):
        l = 0
        self.count += 1
        self.countPacket += 1
        self.destIP.append(element)
        self.responseTime(element)
        if self.count == 50:
            self.start = True
            self.isOK = True
            for i in self.destIP:
                l += 1
                if i not in self.destFrequency:
                    self.destFrequency[i] = 0
                self.destFrequency[i] += 1
            self.findEntropy(self.destFrequency)

            self.destFrequency = {}
            self.destIP = []
            l = 0
            self.count = 0

    def responseTime(self, element):
        self.destIP2.append(element)
        for j in self.destIP2:
            if self.destIP2.count(j) >= 8 and self.flag != 1:
                self.flag = 1
                self.startMarker()
        if len(self.destIP2) == 100:
            self.destIP2 = []

    def startMarker(self):
        start = time.time()
        self.startList.append(start)
        if len(self.startList) == 50:
            #del self.startList[:]	######
            self.startList = []
        return self.startList[0]

    def findEntropy(self, lists):
        l = 50
        entropyList = []
        for k, p in lists.items():
            c = p/float(l)
            c = abs(c)
            entropyList.append(-c * math.log(c, 10))

        self.destEntropy.append(sum(entropyList))

        if(len(self.destEntropy)) == 50:
            print("Sumary: ",self.destEntropy)
            self.destEntropy = []
        self.value = sum(entropyList)


    def __intit__(self):
        pass
