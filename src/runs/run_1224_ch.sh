# dataset, clType, gpuId, spSize, preEpoch1, preEpoch2
sh run.sh ntcir_ch lstm $1 8000 500 jieba
sh run.sh ntcir_ch dan $1 8000 500 jieba
sh run.sh hotel_ch lstm $1 8000 500 jieba
sh run.sh hotel_ch dan $1 8000 500 jieba
