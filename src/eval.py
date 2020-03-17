import argparse
import chainer
from chainer import cuda
from chainer import functions as F
import dill
import dataset
import module
import config
from time import time
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--modelPath',help='model is [dataset, neural_model]')
parser.add_argument('--gpuid', type=int, default=-1)
parser.add_argument('--useChar', action='store_true')
parser.add_argument('--useWord', action='store_true')
parser.add_argument('--useCache', action='store_true')
parser.add_argument('--initLMPath')
args = parser.parse_args()

gpuid = args.gpuid
batchSize = 64
useChar = args.useChar
useWord = args.useWord
useCache = args.useCache

def evaluate(ds,model):
    if gpuid>=0:
        cuda.get_device(gpuid).use()
        model.to_gpu(gpuid)

    segLines = []
    dists = []
    golds = []
    
    scores = []
    for ty in [1,2]:#,2]: # ty=1,2 means valid, test respectively
        indices = list(range(len(ds.labels[ty])))
        batches = module.pack(indices, config.batchSize)

        preds = []
        tags = []


        for batch in batches:
            lines = [ds.data[ty][b] for b in batch]
            ids = [ds.getIdLine(indice=None,
                                     line=line,
                                     useChar=useChar,
                                     useWord=useWord,
                                     useCache=useCache,
                                     sampling=False,
                                     train=False,
                                     unkReplacingRate=0.0) for line in lines]

            charIds = None
            charIds_cpu = None
            wordIds = None

            if useChar:
                charIds_cpu = [idpair[0] for idpair in ids]
                charIds_cpu = [[np.array(word,'i') for word in idLine] for idLine in charIds_cpu]
            if useWord and not useCache:
                wordIds = [idpair[1] for idpair in ids]
                wordIds = [np.array(idLine, 'i') for idLine in wordIds]

            if gpuid>=0:
                if charIds_cpu:
                    charIds = [cuda.to_gpu(idLine) for idLine in charIds_cpu]
                if wordIds:
                    wordIds = cuda.to_gpu(wordIds)
            else:
                charIds = charIds_cpu
                wordIds = wordIds

            # downstream
            zs = model(charIds, wordIds, charIds_cpu)
            preds += np.argmax(zs.data, axis=1).tolist()
       
            zs = F.softmax(zs)
            for i,idLine in enumerate(ids):
                idLine = idLine[0] #char ids
                segLine = '_'.join([''.join(ds.ids2chars(w)) for w in idLine])
                dist = zs[i]
                b = batch[i]
                #print(segLine)
                #print(dist)
                #print(ds.labels[ty][b])

                segLines.append(segLine)
                dists.append(dist)
            golds += [ds.labels[ty][b] for b in batch]

        score = f1_score(ds.labels[ty], preds, average='micro')
        scores.append(score)
        print('valid' if ty==1 else 'test')
        print(score)
        print(classification_report(ds.labels[ty], preds))

    scores_str = '%f\t%f'%(scores[0],scores[1])
    print(score_str)

    return golds, segLines, dists

# trained model
models = dill.load(open(args.modelPath,'rb'))
ds, model = models
with chainer.no_backprop_mode(),chainer.using_config('train', False):
    golds, segLines_train, dists_train = evaluate(ds,model)

# initial model
ds.segmenter.loadModel(args.initLMPath)
with chainer.no_backprop_mode(),chainer.using_config('train', False):
    golds, segLines_init, dists_init = evaluate(ds,model)

#show
for i in range(len(segLines_train)):
    predict_init = np.argmax(dists_init[i].data)
    predict_train = np.argmax(dists_train[i].data)
    #if dists_init[i].data.tolist()!=dists_train[i].data.tolist():
    #if segLines_init[i] != segLines_train[i]:
    if predict_init != predict_train:
        print('gold label:', golds[i])
        print('initial LM', predict_init)
        print(segLines_init[i])
        print(dists_init[i])
        print('trained LM', predict_train)
        print(segLines_train[i])
        print(dists_train[i])
        print('---------------')
