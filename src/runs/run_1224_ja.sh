# dataset, clType, gpuId, spSize, preEpoch1, preEpoch2
sh run.sh ntcir_ja lstm $1 8000 500 mecab
sh run.sh ntcir_ja dan $1 8000 500 mecab
sh run.sh twitter_ja lstm $1 8000 500 mecab
sh run.sh twitter_ja dan $1 8000 500 mecab
