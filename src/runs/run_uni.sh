python manager.py --dataset $1 --segMode uni --clType lstm --useChar --gpuId $2 --preEpoch $3 --hyp $4 --outputPathAffix hyp$4 &
python manager.py --dataset $1 --segMode uni --clType lstm --useChar --sampling --gpuId $2 --preEpoch $3 --hyp $4 --outputPathAffix hyp$4 &
python manager.py --dataset $1 --segMode uni --clType lstm --useChar --uniTrain --gpuId $2 --preEpoch $3 --hyp $4 --outputPathAffix hyp$4 &
wait
