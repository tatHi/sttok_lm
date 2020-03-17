# dataset, clType, gpuId
python manager.py --dataset $1 --useChar --segMode dict --clType $2 --gpuId $3 & 
python manager.py --dataset $1 --useWord --segMode dict --clType $2 --gpuId $3 &
python manager.py --dataset $1 --useChar --useWord --segMode dict --clType $2 --gpuId $3 &
wait
