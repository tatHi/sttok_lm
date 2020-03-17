import config
import numpy as np
cimport numpy as np

# for character
class BigramLM():
    def __init__(self):
        self.unigramDict = {}
        self.bigramDict = {}

        self.unigramCount = None

        self.beta = config.beta
        self.gamma = config.gamma

    def addWord(self, str word):
        cdef int size = len(word)
        cdef int i
        cdef str uni,bi
        cdef dict unigramDict = self.unigramDict
        cdef dict bigramDict = self.bigramDict

        for i in range(size):
            uni = word[i]
            
            if uni in unigramDict:
                unigramDict[uni] += 1
            else:
                unigramDict[uni] = 1

            if i+1<len(word):
                bi = word[i:i+2]
                if bi in bigramDict:
                    bigramDict[bi] += 1
                else:
                    bigramDict[bi] = 1
        
        # reset count
        self.unigramCount = None

    def reduceWord(self, word):
        cdef int size = len(word)
        cdef int i
        cdef str uni,bi
        cdef dict unigramDict = self.unigramDict
        cdef dict bigramDict = self.bigramDict
        for i in range(size):
            uni = word[i]
            unigramDict[uni] -= 1

            if unigramDict[uni] == 0:
                del unigramDict[uni]

            if i+1<len(word):
                bi = word[i:i+2]
                bigramDict[bi] -= 1
                if bigramDict[bi]==0:
                    del bigramDict[bi]

        # reset count
        self.unigramCount = None

    def setBeta(self):
        self.beta = np.average([c for w,c in self.unigramDict.items()])

    def setGamma(self):
        self.gamma = np.average([c for w,c in self.bigramDict.items()])

    def unigramProb(self, str uni):
        if self.unigramCount is None:
            self.setUnigramCount()

        cdef dict unigramDict = self.unigramDict
        cdef double beta = config.beta
        cdef double z = self.unigramCount
        cdef int cnt = unigramDict[uni] if uni in unigramDict else 0
        cdef double pbase = 1/z
        cdef double p = (cnt+beta*pbase)/(z+beta)
        return p

    def bigramProb(self, str bi):
        '''
        beta = 0.01
        p1 = self.bigramDict[bi]/self.unigramDict[bi[0]] if bi in self.bigramDict else 0
        p2 = self.unigramProb(bi[0])*self.unigramProb(bi[1])
        return (1-beta)*p1 + beta*p2
        '''
        cdef dict unigramDict = self.unigramDict
        cdef dict bigramDict = self.bigramDict
        cdef str prev = bi[0]
        cdef str subs = bi[1]
        cdef double gamma = config.gamma
        cdef double y = unigramDict[prev] if prev in unigramDict else 0
        cdef int cnt = bigramDict[bi] if bi in bigramDict else 0
        cdef double pbase = self.unigramProb(prev) * self.unigramProb(subs)

        cdef p = (cnt+gamma*pbase)/(y+gamma)

        return p

    def setUnigramCount(self):
        cdef dict unigramDict = self.unigramDict
        cdef str k
        self.unigramCount = sum([unigramDict[k] for k in unigramDict])

    def wordProb(self, str word, str segMarker=' '):
        # unigram(w0)*prod(bigram(w_i, w_i+1))
        cdef double p
        cdef str bi
        cdef int size = len(word)
        cdef int i
        
        p = self.unigramProb(word[0])
        for i in range(1,size):
            bi = word[i-1:i+1]
            p *= self.bigramProb(bi)
        return p

if __name__ == '__main__':
    b = BigramLM()
    b.addWord('abc')
    b.addWord('ab')
