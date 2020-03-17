import sentencepiece as spm

import sys

inPath = sys.argv[1]
vocSize = int(sys.argv[2])

tmp = '/'.join(inPath.split('/')[:-1])
modelName = '%s/m_%d'%(tmp,vocSize)

spm.SentencePieceTrainer.Train('--input=%s --model_prefix=%s --vocab_size=%d'%(inPath,modelName,vocSize))
sp = spm.SentencePieceProcessor()

