# unigram langage model
# implement- add_words, reduce words, probs
# memory seged line
import numpy as np
cimport numpy as np
import dp
import config
from tqdm import tqdm
import bigram

class UnigramLM:
    def __init__(self):
        self.unigramDict = {}
        self.wordSize = None
        self.segedData = []
        self.charProbDict = {}
        #self.lam = None

        self.bigramLM = bigram.BigramLM()
        
        self.alpha = config.alpha

    def saveModel(self, path):
        import pickle
        pickle.dump((self.unigramDict,self.charProbDict, self.bigramLM), open(path,'wb'))

    def loadModel(self, path):
        import pickle
        self.unigramDict,self.charProbDict, self.bigramLM = pickle.load(open(path,'rb'))

    def addSentence(self, indice):
        # add words in indiced sentence
        self.addWords(self.segedData[indice])

    def reduceSentence(self, indice):
        # reduce words in indiced sentence
        self.reduceWords(self.segedData[indice])

    def addWords(self, list words):
        cdef str word
        cdef dict unigramDict = self.unigramDict

        for word in words:
            if word in unigramDict:
                unigramDict[word] += 1
            else:
                unigramDict[word] = 1

            self.bigramLM.addWord(word)

        self.initializeMemory()

    def reduceWords(self, list words):
        cdef str word
        cdef dict unigramDict = self.unigramDict
        
        for word in words:
            if word in unigramDict:
                unigramDict[word] -= 1
            else:
                print('reduce error: try to reduce words whose count is 0')
                exit()
            
            if unigramDict[word]==0:
                del unigramDict[word]
   
            self.bigramLM.reduceWord(word)

        self.initializeMemory()

    def setAlpha(self):
        cdef:
            int c
            str w
        self.alpha = np.average([c for w,c in self.unigramDict.items()])

    def setWordSize(self):
        cdef str word
        cdef dict unigramDict = self.unigramDict
        self.wordSize = sum([unigramDict[word] for word in unigramDict])

    def getWordProb(self, str word, str segMarker=' '):
        if segMarker in word:
            if word == segMarker:
                return 1
            else:
                return 0

        if self.wordSize is None:
            self.setWordSize()
        
        cdef double alpha = self.alpha
        cdef double wordSize = self.wordSize
        cdef double pc = self.bigramLM.wordProb(word)
        cdef int count = self.unigramDict[word] if word in self.unigramDict else 0
        cdef double prob = (count+alpha*(pc))/(wordSize+alpha)

        if prob == 0:
            print('p(%s)=0'%word)
            exit()

        return prob

    # loadされたパラメータでopt segする
    def setUnigramWithLoadParam(self, data):
        self.data = data
        self.segedData = self.optSegment(data)
        self.unigramDict = {}
        for i in range(len(self.data)):
            self.addSentence(i)

    def setInitialUnigram(self, data, padding=False, segMarker=' '):
        self.data = data

        # set words with random segmentation
        ws = []
        for line in data:
            if padding:
                # ignore padding words
                line = line[1:-1]
            w = ''
            segedLine = []
            for i,c in enumerate(line):
                if c==segMarker:
                    if w!='':
                        segedLine.append(w)
                        w = ''
                    segedLine.append(c)
                else:
                    w += c
                    if i==len(line)-1 or np.random.rand() > 0.5:
                        segedLine.append(w)
                        w = ''
            ws += segedLine
            self.segedData.append(segedLine)
        self.addWords(ws)

        # initialize char dict
        charDict = {}
        charSize = 0
        for line in data:
            for c in line:
                if c in charDict:
                    charDict[c] += 1
                else:
                    charDict[c] = 1
            charSize += len(line)
        
        for c in charDict:
            self.charProbDict[c] = charDict[c]/charSize
        
        self.charProbDict['<UNK>'] = 1/charSize # UNK

        # initialize id2char dicts
        #self.char2id, self.id2char = dictMaker.getDict(data)
    
    def initializeMemory(self):
        self.wordSize = None
        #self.lam = None

    def train(self, list data, int maxEpoch, str outPath):
        cdef:
            int epoch,i 
            np.ndarray indices
            str line
            list segedLine

        print('set random segmentation')
        self.setInitialUnigram(data,padding=False) 

        for epoch in range(maxEpoch):    
            indices = np.random.permutation(len(self.segedData))
            for i in tqdm(indices):
                self.reduceSentence(i)
    
                line = self.data[i]
                segedLine = dp.segment(line,self,config.maxLength,sampling=True)
                self.segedData[i] = segedLine
                self.addSentence(i)

            print('epoch %d done'%epoch)
            for i in range(min(10, len(self.segedData))):
                print('_'.join(self.segedData[i]))

            if (epoch+1)%100==0:
                self.saveModel(outPath.replace('.model','_%d.model'%(epoch+1)))

    def optSegment(self, data):
        optData = []
        print('opt segment...')
        for line in tqdm(data):
            segedLine = dp.segment(line,self,config.maxLength,sampling=False)
            optData.append(segedLine)
        return optData

    def encodeSentence(self, line, sampling=False, segMarker=' '):
        segedLine = dp.segment(line,self,config.maxLength,sampling=False)
        
        # Encode時はsegMarkerを削除する
        segedLine = [w for w in segedLine if w!=segMarker]

        return segedLine

    # encode時にencodeAndUpdateできるようにする
    def encodeAndUpdate(self, i, segMarker=' '):
        if not config.reduceAfterSampling:
            # reduce
            self.reduceSentence(i)
        
        # sampling
        line = ''.join(self.segedData[i])
        segedLine = dp.segment(line,self,config.maxLength,sampling=True)
        
        # update
        if config.reduceAfterSampling:
            # reduce
            self.reduceSentence(i)

        self.segedData[i] = segedLine
        self.addSentence(i)

        # Encode時はsegMarkerを削除する
        segedLine = [w for w in segedLine if w!=segMarker]
        
        return segedLine

    # wrap
    def EncodeAsPieces(self, line):
        return self.encodeSentence(line, sampling=False)

    def SampleEncodeAsPieces(self,line,alpha=None,nbest_size=None):
        # alpha and nbest_size args are dammy
        return self.encodeSentence(line,sampling=True)

    def setLMfromText(self, path):
        # path to text data splited with ' '
        data = [line.strip().split() for line in open(path)] 
        words = [word for line in data for word in line] 
        self.addWords(words)


if __name__ == '__main__':
    import sys
    path = sys.argv[1]
    uni = UnigramLM()
    data = [line.strip() for line in open(path)]
    modelPath = path.replace('text.txt','unigram_bigram.model')
    uni.train(data, 100, modelPath)
