#dataset, segMode, gpuId

dataset=ntcir_ja
sh run_dict.sh $dataset lstm 0 
sh run_dict.sh $dataset dan 0
dataset=ntcir_ch
sh run_dict.sh $dataset lstm 0 
sh run_dict.sh $dataset dan 0
dataset=ntcir_en
sh run_dict.sh $dataset lstm 0 
sh run_dict.sh $dataset dan 0
dataset=twitter_ja
sh run_dict.sh $dataset lstm 0 
sh run_dict.sh $dataset dan 0
dataset=hotel_ch
sh run_dict.sh $dataset lstm 0 
sh run_dict.sh $dataset dan 0
dataset=twitter_en
sh run_dict.sh $dataset lstm 0 
sh run_dict.sh $dataset dan 0
