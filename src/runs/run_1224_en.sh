# dataset, clType, gpuId, spSize, preEpoch1, preEpoch2
sh run.sh ntcir_en lstm $1 6000 500 original
sh run.sh ntcir_en dan $1 6000 500 original
sh run.sh twitter_en lstm $1 8000 500 original
sh run.sh twitter_en dan $1 8000 500 original
