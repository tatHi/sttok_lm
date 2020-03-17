# dataset, clType, gpuId, spSize, preEpoch1(supervised), preEpoch2(unsupervised)

# char and sp-word
#python manager.py --dataset $1 --segMode char --clType $2 --useWord --gpuId $3 &
#python manager.py --dataset $1 --segMode sp --clType $2 --useWord --gpuId $3 --spSize $4 &
#python manager.py --dataset $1 --segMode sp --clType $2 --useWord --sampling --gpuId $3 --spSize $4 &

#wait
# sp-char
#python manager.py --dataset $1 --segMode sp --clType $2 --useChar --gpuId $3 --spSize $4 &
#python manager.py --dataset $1 --segMode sp --clType $2 --useChar --sampling --gpuId $3 --spSize $4 &
#wait

# sp-char-word
#python manager.py --dataset $1 --segMode sp --clType $2 --useChar --useWord --gpuId $3 --spSize $4 &
#python manager.py --dataset $1 --segMode sp --clType $2 --useChar --useWord --sampling --gpuId $3 --spSize $4 &
#wait

# uni-char supervised (dict)
python manager.py --dataset $1 --segMode uni --clType $2 --useChar --gpuId $3 --preEpoch $5 --outputPathAffix $5\_ &
python manager.py --dataset $1 --segMode uni --clType $2 --useChar --sampling --gpuId $3 --preEpoch $5 --outputPathAffix $5\_ &
python manager.py --dataset $1 --segMode uni --clType $2 --useChar --uniTrain --gpuId $3 --preEpoch $5 --outputPathAffix $5\_ &
wait

# uni-char unsupervised
python manager.py --dataset $1 --segMode uni --clType $2 --useChar --gpuId $3 --preEpoch $6 --outputPathAffix $6\_ &
python manager.py --dataset $1 --segMode uni --clType $2 --useChar --sampling --gpuId $3 --preEpoch $6 --outputPathAffix $6\_ &
python manager.py --dataset $1 --segMode uni --clType $2 --useChar --uniTrain --gpuId $3 --preEpoch $6 --outputPathAffix $6\_ &
wait

# uni-char-word supervised
python manager.py --dataset $1 --segMode uni --clType $2 --useChar --useWord --gpuId $3 --preEpoch $5 --outputPathAffix $5\_ &
python manager.py --dataset $1 --segMode uni --clType $2 --useChar --useWord --sampling --gpuId $3 --preEpoch $5 --outputPathAffix $5\_ &
python manager.py --dataset $1 --segMode uni --clType $2 --useChar --useWord --uniTrain --gpuId $3 --preEpoch $5 --outputPathAffix $5\_ &
wait

# uni-char-word unsupervised
python manager.py --dataset $1 --segMode uni --clType $2 --useChar --useWord --gpuId $3 --preEpoch $6 --outputPathAffix $6\_ &
python manager.py --dataset $1 --segMode uni --clType $2 --useChar --useWord --sampling --gpuId $3 --preEpoch $6 --outputPathAffix $6\_ &
python manager.py --dataset $1 --segMode uni --clType $2 --useChar --useWord --uniTrain --gpuId $3 --preEpoch $6 --outputPathAffix $6\_ &
wait
