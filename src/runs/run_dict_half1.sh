#dataset, segMode, gpuId

dataset=ntcir_ja
sh run_dict.sh $dataset lstm $1 
sh run_dict.sh $dataset dan $1
dataset=ntcir_ch
sh run_dict.sh $dataset lstm $1 
sh run_dict.sh $dataset dan $1
dataset=twitter_en
sh run_dict.sh $dataset lstm $1 
sh run_dict.sh $dataset dan $1
