import sys
subwordPath = sys.argv[1]
spPath = sys.argv[2]
unspPath = sys.argv[3]

# sentencePiece
vs = [line.strip().split('\t')[0] for line in open(subwordPath)]
print('sentencepeace average:', sum([len(v) for v in vs])/len(vs))

import unigram
u = unigram.UnigramLM()
u.loadModel(spPath)
vs = [v for v in u.unigramDict]
print('proposed dict average:', sum([len(v) for v in vs])/len(vs))

u = unigram.UnigramLM()
u.loadModel(unspPath)
vs = [v for v in u.unigramDict]
print('proposed unsp average:', sum([len(v) for v in vs])/len(vs))
