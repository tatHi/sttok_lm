# sh run_uni_ch.sh hotel_ch dan 2 8000 500 jieba

python manager.py --dataset $1 --segMode uni --clType $2 --useChar --gpuId $3 --preEpoch $5 --outputPathAffix $5\_ &
python manager.py --dataset $1 --segMode uni --clType $2 --useChar --sampling --gpuId $3 --preEpoch $5 --outputPathAffix $5\_ &
wait

python manager.py --dataset $1 --segMode uni --clType $2 --useChar --uniTrain --gpuId $3 --preEpoch $5 --outputPathAffix $5\_ &
python manager.py --dataset $1 --segMode uni --clType $2 --useChar --gpuId $3 --preEpoch $6 --outputPathAffix $6\_ &
wait
python manager.py --dataset $1 --segMode uni --clType $2 --useChar --sampling --gpuId $3 --preEpoch $6 --outputPathAffix $6\_ &
python manager.py --dataset $1 --segMode uni --clType $2 --useChar --uniTrain --gpuId $3 --preEpoch $6 --outputPathAffix $6\_ &
wait
