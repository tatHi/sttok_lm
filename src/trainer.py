import config
import dataset
import segDataset
import unigram
import dp

import model as M
import multiEmbed as EM
import clModel as CM

import module
import numpy as np
import chainer
from chainer import optimizers
from chainer import cuda
from chainer import functions as F
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import logger
from tqdm import tqdm
from time import time
import dill

'''
segMode = 'uni'#'sp'
uniTrain = True
'''

class Trainer:
    #        useCache, clType, sampling, uniTrain, modelPath, classSize, gpuid, smoothingHyp):        
    def __init__(self, **kwargs):
        textPathes = kwargs['textPathes']
        labelPathes = kwargs['labelPathes']
        loggerPathes = kwargs['loggerPathes']
        self.dumpPath = kwargs['dumpPath']
        segMode = kwargs['segMode']
        useChar = kwargs['useChar']
        useWord = kwargs['useWord']
        useCache = kwargs['useCache']
        clType = kwargs['clType']
        sampling = kwargs['sampling']
        uniTrain = kwargs['uniTrain']
        modelPath = kwargs['modelPath']
        gpuid = kwargs['gpuid']

        self.gpuid = gpuid
        if gpuid >= 0:
            from chainer import cuda
            self.xp = cuda.cupy
            cuda.get_device(gpuid).use()
        else:
            import numpy
            self.xp = numpy

        self.ds = dataset.Dataset(textPathes=textPathes,
                                   labelPathes=labelPathes,
                                   segMode=segMode,
                                   modelPath=modelPath,
                                   padding=False)

        # set smoothingHyp
        if segMode=='uni':
            self.ds.segmenter.alpha = config.alpha
            self.ds.segmenter.bigramLM.beta = config.beta
            self.ds.segmenter.bigramLM.gamma = config.gamma

        charEmbedSize = 30
        wordEmbedSize = 100
        hidSize = 200
        outSize = self.ds.classSize

        self.useChar = useChar
        self.useWord = useWord
        self.useCache = useCache
        self.sampling = sampling
        self.uniTrain = uniTrain

        charVocSize = len(self.ds.id2char) if useChar else None
        wordVocSize = None
        if segMode=='char':
            wordVocSize = len(self.ds.id2char)
        elif segMode=='dict':
            wordVocSize = len(self.ds.id2word) if useWord else None
        else:
            if useCache:
                wordVocSize = kwargs['spSize']
            elif segMode=='sp':
                wordVocSize = self.ds.segmenter.get_piece_size() if useWord else None

        embedModel = EM.MultiEmbed(charVocSize=charVocSize,
                                   charEmbedSize=charVocSize,
                                   wordVocSize=wordVocSize,
                                   wordEmbedSize=wordEmbedSize,
                                   useCache=useCache,
                                   gpuid=gpuid)

        if clType=='dan':
            clModel = CM.DANModel(wordEmbedSize, hidSize, outSize)
        elif clType=='bilstm':
            clModel = CM.BiLSTMModel(wordEmbedSize, hidSize, outSize)
        elif clType=='lstm':
            clModel = CM.LSTMModel(wordEmbedSize, hidSize, outSize)
        elif clType=='attn':
            clModel = CM.AttentiveModel(wordEmbedSize, outSize)
        self.model = M.Model(embedModel, clModel)

        self.eval_logger = logger.Logger(loggerPathes[0])
        self.loss_logger = logger.Logger(loggerPathes[1])

    def epochProcess(self, epoch, opt):
        print(epoch)

        indices = np.random.permutation(len(self.ds.idData[0]))
        batches = module.pack(indices, config.batchSize)
        st = time()

        for i,batch in enumerate(batches):
            self.model.cleargrads()
            ids = [self.ds.getIdLine(indice=(0,b), # indice=0 means train dataset
                                     line=None,
                                     useChar=self.useChar,
                                     useWord=self.useWord,
                                     useCache=self.useCache,
                                     sampling=self.sampling,
                                     train=self.uniTrain,
                                     unkReplacingRate=0.0) for b in batch]

            charIds = None
            charIds_cpu = None
            wordIds = None

            if self.useChar:
                charIds_cpu = [idpair[0] for idpair in ids]
                charIds_cpu = [[np.array(word,'i') for word in idLine] for idLine in charIds_cpu]
            if self.useWord and not self.useCache:
                wordIds = [idpair[1] for idpair in ids]
                wordIds = [np.array(idLine, 'i') for idLine in wordIds]
 
            ts_cpu = [self.ds.labels[0][b] for b in batch]
            ts_cpu = np.array(ts_cpu,'i')

            if self.gpuid>=0:
                if charIds_cpu:
                    charIds = [cuda.to_gpu(idLine) for idLine in charIds_cpu]
                if wordIds:
                    wordIds = cuda.to_gpu(wordIds)
                ts = cuda.to_gpu(ts_cpu)
            else:
                charIds = charIds_cpu
                wordIds = wordIds
                ts = ts_cpu
          
            loss = self.model.getLoss(charIds, wordIds, ts, charIds_cpu)
            
            if self.xp.isnan(loss.data):
                print('nan')
                exit()

            print('epoch:%d batch:(%d/%d) loss:%f'%(epoch, i+1, len(batches), loss.data.tolist()))
        
            #opt.setup(self.model)
            self.model.cleargrads()
            loss.backward()
            opt.update()    
            self.loss_logger.write(loss.data)

        processTime = time()-st
        print('time:',processTime)
        print('time/sent', processTime/len(indices))


    def train(self):
        # initial unigram training
        # self.uni_train.train(10)
        if self.gpuid>=0:
            cuda.get_device(self.gpuid).use()
            self.model.to_gpu(self.gpuid)

        opt = optimizers.Adam()
        #opt = optimizers.SGD(lr=0.1)
        opt.setup(self.model)

        maxVaridScore = 0
        for epoch in range(config.maxEpoch):
            
            self.epochProcess(epoch, opt)
            
            # evaluation
            with chainer.no_backprop_mode(),chainer.using_config('train', False):
                validScore, testScore = self.evaluate(epoch)
            
            # save model if valid score is maximum
            if maxVaridScore<=validScore and self.dumpPath:
                if self.gpuid>=0:
                    self.model.to_cpu()
                maxVaridScore = validScore
                dill.dump((self.ds, self.model),open(self.dumpPath,'wb'))
                if self.gpuid>=0:
                    self.model.to_gpu(self.gpuid)

    def evaluate(self, epoch):
        if self.gpuid>=0:
            cuda.get_device(self.gpuid).use()
            self.model.to_gpu(self.gpuid)

        scores = []
        for ty in [1,2]: # ty=1,2 means valid, test respectively
            indices = list(range(len(self.ds.labels[ty])))
            batches = module.pack(indices, config.batchSize)

            preds = []
            tags = []
            
            print('eval...')
            startTime = time()
            for batch in tqdm(batches):
                lines = [self.ds.data[ty][b] for b in batch]
                ids = [self.ds.getIdLine(indice=None,
                                         line=line,
                                         useChar=self.useChar,
                                         useWord=self.useWord,
                                         useCache=self.useCache,
                                         sampling=False,
                                         train=False,
                                         unkReplacingRate=0.0) for line in lines]

                charIds = None
                charIds_cpu = None
                wordIds = None

                if self.useChar:
                    charIds_cpu = [idpair[0] for idpair in ids]
                    charIds_cpu = [[np.array(word,'i') for word in idLine] for idLine in charIds_cpu]
                if self.useWord and not self.useCache:
                    wordIds = [idpair[1] for idpair in ids]
                    wordIds = [np.array(idLine, 'i') for idLine in wordIds]

                if self.gpuid>=0:
                    if charIds_cpu:
                        charIds = [cuda.to_gpu(idLine) for idLine in charIds_cpu]
                    if wordIds:
                        wordIds = cuda.to_gpu(wordIds)
                else:
                    charIds = charIds_cpu
                    wordIds = wordIds

                # downstream
                zs = self.model(charIds, wordIds, charIds_cpu)
                preds += np.argmax(zs.data, axis=1).tolist()
            
            print('time:',time()-startTime)

            score = f1_score(self.ds.labels[ty], preds, average='micro')
            scores.append(score)
            print('valid' if ty==1 else 'test')
            print(score)
            print(classification_report(self.ds.labels[ty], preds))

        scores_str = '%f\t%f'%(scores[0],scores[1])
        self.eval_logger.write(scores_str)

        return scores

def segmentIdLine(idLine, seg):
    neoIdLine = []
    flag = 0
    for s in seg:
        neoIdLine.append(idLine[flag:flag+s])
        flag += s
    return neoIdLine

if __name__ == '__main__':
    t = Trainer()
    t.train()
