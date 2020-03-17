# run_uni_reduce_after.sh ntcir_ja lstm 0 8000 500 mecab
python manager.py --dataset $1 --segMode uni --clType $2 --useChar --uniTrain --gpuId $3 --preEpoch $5 --outputPathAffix $5\_ --reduceAfterSampling &
python manager.py --dataset $1 --segMode uni --clType $2 --useChar --uniTrain --gpuId $3 --preEpoch $6 --outputPathAffix $6\_ --reduceAfterSampling &
