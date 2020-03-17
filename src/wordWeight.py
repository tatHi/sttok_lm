from chainer import Chain
from chainer import links as L
from chainer import functions as F

class WordWeight(Chain):
    def __init__(self, wordVecSize):
        super().__init__()
        linear = L.Linear(wordVecSize, 1)
        self.add_link('linear',linear)
        self.wordVecSize = wordVecSize

    def forward(self, wordVecs):
        weights = F.softmax(self.linear(wordVecs),axis=0)
        weightedVecs = wordVecs * F.tile(weights,(1,self.wordVecSize))
        return weightedVecs

if __name__ == '__main__':
    import numpy as np
    wvs = 100
    voc = 10000

    from time import time

    st = time()
    ems = np.random.rand(voc,wvs).astype('f')
    ww = WordWeight(wvs)
    ww(ems)
    print(time()-st)
