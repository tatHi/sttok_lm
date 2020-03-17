# sh run_uni_hyp.sh ntcir_ja lstm 5 1 500 mecab
python manager.py --dataset $1 --segMode uni --clType $2 --useChar --useWord --uniTrain --gpuId $4 --preEpoch $5 --smoothingHyp $3 --outputPathAffix $5\_ &
python manager.py --dataset $1 --segMode uni --clType $2 --useChar --useWord --uniTrain --gpuId $4 --preEpoch $6 --smoothingHyp $3 --outputPathAffix $6\_ &

#python manager.py --dataset $1 --useChar --useWord --uniTrain --segMode uni --clType $2 --preEpoch 500 --gpuId $4 --outputPathAffix 500\_ &
#python manager.py --dataset $1 --useChar --useWord --uniTrain --segMode uni --clType $2 --preEpoch $3 --gpuId $4 --outputPathAffix $3\_ &
