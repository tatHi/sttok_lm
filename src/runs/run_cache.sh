# sh run_cache.sh ntcir_ja lstm mecab 0
python manager.py --dataset $1 --useChar --useWord --uniTrain --segMode uni --clType $2 --preEpoch 500 --gpuId $4 --outputPathAffix 500\_ &
python manager.py --dataset $1 --useChar --useWord --uniTrain --segMode uni --clType $2 --preEpoch $3 --gpuId $4 --outputPathAffix $3\_ &
wait
