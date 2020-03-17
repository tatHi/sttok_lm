from unigram import UnigramLM
import sys
import config

path = sys.argv[1]

suffix = '_hyp%d'%config.hyp

uni = UnigramLM()
data = [line.strip() for line in open(path)]
modelPath = path.replace('text.txt','unigram_bigram%s.model'%suffix)
print(modelPath)
uni.train(data, 1000, modelPath)

