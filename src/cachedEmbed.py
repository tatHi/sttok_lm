import chainer
from chainer import Chain
from chainer import functions as F
from chainer import links as L
import config

class CachedEmbed(Chain):
    def __init__(self, cacheSize, embedSize, gpuId):
        super().__init__()

        self.wordCache = []
        self.word2id = {}
        vecTable = L.EmbedID(cacheSize, embedSize)
        self.add_link('vecTable',vecTable)

        self.initializeCache(cacheSize)
        
        if gpuId>=0:
            from chainer import cuda
            self._xp = cuda.cupy
        else:
            import numpy
            self._xp = numpy

    def initializeCache(self, cacheSize):
        self.wordCache = [i for i in range(cacheSize)]
        self.word2id = {i:i for i in range(cacheSize)}

    def forward(self, idLines, wordVecs):
        # idLine: [[1,2,3],[4,5]]
        # wordVec: vector of size (len(idLine),self.embedSize)
        words = getWordList(idLines)
        wordVecs = F.vstack(wordVecs)

        #print('words')
        #print(words)
        #print('vecs')
        #print(vecs)

        if chainer.config.train:
            # updation
            ids = [] # for extructing vecs from lookup table
            for word,vec in zip(words,wordVecs):
                #print('---')
                #print('cache:',self.wordCache)
                #print('w2i:',self.word2id)
                #print('ids:',ids)
                #print('word:',word)

                if word in self.word2id:
                    # キャッシュにある場合は、一度削除してあとで先頭につけ足す
                    idx = self.word2id[word]
                    if idx not in ids:
                        self.wordCache.remove(word)
                    ids.append(idx)
                else:
                    # not in cache:
                    # ひとつポップして対応するIDを割り当てる。
                    # tableをvecで初期化
                    popedWord = self.wordCache.pop()
                    
                    # popされた単語のIDを取り出して、削除
                    idx = self.word2id[popedWord]
                    del self.word2id[popedWord]

                    self.word2id[word]=idx
                    self.vecTable.W.data[idx]=vec.data
                    ids.append(idx)

            # update word cache by concatenating set(ids) with the head of word cache
            self.wordCache = list(set(words))+self.wordCache

            #print(ids)

            vecs = self.vecTable(self._xp.array(ids,'i'))
            #vecs = self.vecTable.W[ids,]
            ###
        else:
            # without updation
            reps = []
            for word, vec in zip(words,wordVecs):
                if word in self.word2id:
                    reps.append(self.vecTable.W[self.word2id[word]])
                else:
                    reps.append(vec)
            vecs = F.vstack(reps)

        #print(self.vecTable.W[ids,])
        #print(vecs)
       
        splitedVecs = []
        flag = 0
        for idLine in idLines:
            splitedVecs.append(vecs[flag:flag+len(idLine)])
            flag += len(idLine)
        return splitedVecs

        # if the same words with different vector are included in a single batch and it is not in cache,
        # add the first vector as an initial representation and ignore the others.
        # This happens when using LSTM-minus like embedding in previous provess.
        # It is expected to average vectors of the same words in a single batch to initialize its vector.


def getWordList(idLines):
    # convert batched idLines into flatten tuples
    words = [tuple(word) for idLine in idLines for word in idLine]
    return words

if __name__ == '__main__':
    config.setGPUID(-1)
    import numpy as np
    charIdLines1 = [[np.array([1,2,3],'i'),np.array([2,],'i'),np.array([1,2,3,4],'i')],
                    [np.array([1,2],'i'),np.array([1,2,3],'i')]] 
    charIdLines2 = [[np.array([1,2],'i'),np.array([3,],'i')],
                    [np.array([3,2],'i'),np.array([1,2,3],'i')]] 
    wordVecs1 = [np.random.rand(3,3).astype('f'), np.random.rand(2,3).astype('f')]
    wordVecs2 = [np.random.rand(2,3).astype('f'), np.random.rand(2,3).astype('f')]

    ce = CachedEmbed(5, 3, -1)
    ce(charIdLines1, wordVecs1)
    ce(charIdLines2, wordVecs2)
