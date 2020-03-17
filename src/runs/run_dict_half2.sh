#dataset, segMode, gpuId
#dataset=ntcir_en
#sh run_dict.sh $dataset lstm $1
#sh run_dict.sh $dataset dan $1
dataset=twitter_ja
sh run_dict.sh $dataset lstm $1
sh run_dict.sh $dataset dan $1
dataset=hotel_ch
sh run_dict.sh $dataset lstm $1
sh run_dict.sh $dataset dan $1
