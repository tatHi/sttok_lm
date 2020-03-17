from collections import defaultdict
import dictMaker
import pickle

class Dataset:
    def __init__(self, textPathes, labelPathes=None, segMode=None, modelPath=None, dictPath=None, padding=False):
        # given multiple text pathes [train, valid, test, +]
       
        ### parse data ###
        self.data = [[line.strip() for line in open(textPath)] for textPath in textPathes]

        ### set segmenter ###
        if segMode=='uni':
            import unigram
            self.segmenter = unigram.UnigramLM()
            if modelPath:
                self.segmenter.loadModel(modelPath)
                self.segmenter.setUnigramWithLoadParam(self.data[0])
            else:
                # initialize with data[0] (= train data)
                self.segmenter.setInitialUnigram(self.data[0])

        elif segMode=='sp':
            import sentencepiece as spm
            self.segmenter = spm.SentencePieceProcessor()
            print(modelPath)
            self.segmenter.Load(modelPath)
        elif segMode=='char':
            pass

        self.segMode = segMode

        ### set char dict ###
        self.char2id, self.id2char = dictMaker.getDict(self.data[0])
       
        ### set word dict when segMode is dict###
        if segMode=='dict':
            segedData = [line.split() for line in self.data[0]]
            self.word2id, self.id2word = dictMaker.getDict(segedData)

        ### set label ###
        if labelPathes:
            self.labels = [self.getLabel(labelPath) for labelPath in labelPathes]
            self.classSize = len(set(self.labels[0]))
            print('class size:', self.classSize)

        self.idData = [self.getIdData(d, padding) for d in self.data]

        ### set unkId ###
        self.unkId_char = None
        self.unkId_word = None
        if segMode=='uni':
            self.unkId_char = self.char2id['<UNK>']
        elif segMode=='sp':
            self.unkId_char = self.char2id['<UNK>']
            self.unkId_word = self.segmenter.unk_id()
        elif segMode=='char':
            self.unkId_word = self.char2id['<UNK>']
        elif segMode=='dict':
            self.unkId_char = self.char2id['<UNK>']
            self.unkId_word = self.word2id['<UNK>']

    def getLabel(self, path):
        print(path)
        data = [int(line.strip()) for line in open(path) if line.strip()]
        print('label size:', len(data))
        return data

    def getIdData(self, data, padding):
        idData = [self.chars2ids(line) for line in data]
        
        if padding:
            b = [self.char2id[bos]]
            e = [self.char2id[eos]]
            idData = [tuple(b+idLine+e) for idLine in idData]
        else:
            idData = [tuple(idLine) for idLine in idData]

        return idData

    def encodeAsIdLine(self, line, sampling=False):
        if sampling:
            words = self.segmenter.SampleEncodeAsPieces(line,alpha=0.5,nbest_size=-1)
        else:
            words = self.segmenter.EncodeAsPieces(line)
        line = [tuple(self.chars2ids(word)) for word in words]
        return line 

    def getIdLineAndTrain(self, indice):
        words = self.segmenter.encodeAndUpdate(indice)
        line = [tuple(self.chars2ids(word)) for word in words]
        return line
    
    def getIdLine(self, indice=None, line=None, useChar=False, useWord=False, useCache=False,
                  sampling=None, train=None, unkReplacingRate=None):
        if line is None:
            assert indice is not None, 'indice must be given when line is not specified'
            line = self.data[indice[0]][indice[1]]

        wordIds = None
        charIds = None

        if self.segMode=='char':
            if indice:
                assert len(indice) == 2, 'indice should be tuple of size 2 (data-id and line-id)'
                wordIds = self.idData[indice[0]][indice[1]] 
            else:
                wordIds = self.chars2ids(line)
        elif self.segMode=='sp' or self.segMode=='uni':
            if train:
                charIds = self.getIdLineAndTrain(indice[1])
            else:
                if sampling:
                    line_str = self.segmenter.SampleEncodeAsPieces(line,alpha=0.5,nbest_size=-1)
                else:
                    line_str = self.segmenter.EncodeAsPieces(line)

                if useChar:
                    charIds = [tuple(self.chars2ids(word)) for word in line_str]
                if useWord and not useCache:
                    wordIds = [self.segmenter.piece_to_id(word) for word in line_str]
        elif self.segMode=='dict':
            # mecabなどでの分割の場合
            if indice:
                line = self.data[indice[0]][indice[1]]
            charIds = [self.chars2ids(word) for word in line.split()]
            wordIds = self.words2ids(line.split())
      
        if not useCache:
            if useChar and unkReplacingRate>0:
                charIds = [replaceUnk(charId,self.unkId_char,unkReplacingRate) for charId in charIds]
            if useWord and unkReplacingRate>0:
                wordIds = replaceUnk(wordIds,self.unkId_word,unkReplacingRate)

        return charIds, wordIds

    def chars2ids(self, chars):
        return [self.char2id[c] if c in self.char2id else self.char2id['<UNK>'] for c in chars]

    def ids2chars(self, ids):
        return [self.id2char[i] for i in ids]

    # segMode==dictの時のみ呼ばれる
    def words2ids(self, words):
        return [self.word2id[w] if w in self.word2id else self.word2id['<UNK>'] for w in words]

    def ids2words(self, ids):
        return [self.id2word[i] for i in ids]

import random
def replaceUnk(ids, unkId, rate=0.1):
    neoids = [unkId if random.random()<rate else i for i in ids]
    return neoids

if __name__ == '__main__':
    ds = Dataset('../data/ntcir_train_text.txt')
    print(ds.idData[:10])
    print(ds.vocab[:10])
