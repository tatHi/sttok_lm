from chainer import Chain
from chainer import links as L
from chainer import functions as F

class CompEmbed(Chain):
    def __init__(self, charVocSize, charEmbedSize, wordEmbedSize):
        super().__init__()
        embed = L.EmbedID(charVocSize, charEmbedSize)
        lstm = L.NStepLSTM(1,charEmbedSize, wordEmbedSize, dropout=0.)
        
        self.add_link('embed',embed)
        self.add_link('lstm',lstm)

    def forward_(self, idLines):
        hys = []
        for words in idLines:
            # words: [[1,2,3],[2,23],[1,2,3,4,5]]
            ems = [self.embed(word) for word in words]
            _,hy,_ = self.lstm(None,None,ems)
            hys.append(hy[0])
        return hys

    # faster virsion
    def forward(self, idLines):
        words = [w for idLine in idLines for w in idLine]
        ems = [self.embed(word) for word in words]
        _,hy,_ = self.lstm(None,None,ems)
        hy = hy[0]

        # resegment
        hys = []
        flag = 0
        for idLine in idLines:
            size = len(idLine)
            hys.append(hy[flag:flag+size])
            flag += size
        return hys

if __name__ == '__main__':
    import numpy as np
    import random
    from time import time
    idLines = [[np.array([1,2,3],'i'),np.array([1,2,3,4,5,6],'i')],
             [np.array([1,3],'i'),np.array([1,2,5,6],'i')]]
    idLines = [[np.array([random.randint(0,9) for k in range(random.randint(1,8))],'i') for j in range(random.randint(1,20))] for i in range(100)]
    c2w = CompEmbed(10000,30,100)

    st = time()
    #print(c2w.forward_(idLines))
    c2w.forward_(idLines)
    print(time()-st)

    # faster
    st = time()
    #print(c2w.forward(idLines))
    c2w.forward(idLines)
    print(time()-st)
