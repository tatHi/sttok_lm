import numpy as np
cimport numpy as np
from time import time
import config
# unigram dp
# backward probability

def segment(str sentence, uniLM, int maxL, bint sampling, str segMarker=' '):
    cdef:
        list lines, segedLine, ls
        int i, flag, l
        str line
        np.ndarray alpha
    
    lines = sentence.split(segMarker)
    segedLine = []
    
    for i,line in enumerate(lines):
        if line=='':
            continue
        if 0<i:
            segedLine.append(segMarker)
        alpha = unigramDP(line, uniLM, maxL)

        ls = unigramSampling(alpha, not sampling)
        
        flag = 0
        for l in ls:
            segedLine.append(line[flag:flag+l])
            flag += l
        
    return segedLine

cdef np.ndarray unigramDP(str line, uniLM, int maxL):
    cdef:
        list alpha
        int size,i,j,k
        str w
        double p, prevs

    size = len(line)
    alpha = np.zeros((size,maxL)).tolist()
    
    for i in reversed(range(size)):
        for j in range(min(size-i, maxL)):
            w = line[i:i+j+1]
            p = uniLM.getWordProb(w)
            if size-i-1>j:
                prevs = sum([alpha[i+j+1][k] for k in range(min(size-(i+j+1),maxL))])
                p *= prevs
            alpha[i][j] = p

    return np.array(alpha)

cdef list unigramSampling(np.ndarray alpha, bint opt=False):
    cdef:
        list indiceList
        int flag, size, indice
        np.ndarray dist

    size = alpha.shape[0]

    indiceList = []
    flag = 0
    while flag<size:
        sa = sum(alpha[flag])
        if sa==0:
            indice = 0
        else:
            dist = alpha[flag]/sa
            if opt:
                indice = np.argmax(dist)
            else:
                indice = np.random.choice(len(dist),1,p=dist)[0]


        indiceList.append(indice+1)
        flag += indiceList[-1]
    return indiceList

if __name__ == '__main__':
    import unigram
    u = unigram.UnigramLM()
    ws = ['a','a','a','b','b','b','b','c','d','d','ab','bc','bc','cd','bcd']
    u.addWords(ws)
    line = 'abcd'
    unigramDP(line, u, maxL=3)
    
