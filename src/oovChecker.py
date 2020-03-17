import unigram
import argparse
from tqdm import tqdm
### argparse ###
parser = argparse.ArgumentParser()
parser.add_argument('--trainData')
parser.add_argument('--testData')
parser.add_argument('--initialLM')
parser.add_argument('--cacheSize', type=int)
parser.add_argument('--epoch', type=int, default=30)
args = parser.parse_args()

### cache ###
class Cache:
    def __init__(self):
        self.queue = []
        self.allVocab = set()

    def update(self, w):
        if w in self.queue:
            self.queue.remove(w)
        else:
            if len(self.queue) >= args.cacheSize:
                self.queue.pop()
        self.queue.insert(0, w)

        if len(self.queue) > args.cacheSize:
            print('ERROR')
            exit()

        self.allVocab.add(w)

trainData = [line.strip() for line in open(args.trainData)]
testData = [line.strip() for line in open(args.testData)]

def checkOOV(uni, cache):
    optTest = u.optSegment(testData)
    voc = {w for line in optTest for w in line}
    lmOOV = voc - set(u.unigramDict.keys())
    cacheOOV = voc - set(cache.queue)

    vocSize = len(voc)

    print('lm oov rate:      %d/%d=%f'%(len(lmOOV),vocSize,len(lmOOV)/vocSize))
    print('caceh oov rate:   %d/%d=%f'%(len(cacheOOV),vocSize,len(cacheOOV)/vocSize))
    print('total vocab size: %d'%len(c.allVocab))

u = unigram.UnigramLM()
u.loadModel(args.initialLM)
u.setUnigramWithLoadParam(trainData)

c = Cache()

for ep in range(args.epoch):
    print('train')
    for i in tqdm(range(len(trainData))):
        s = u.encodeAndUpdate(i)
        for w in s:
            c.update(w)
    checkOOV(u,c)

