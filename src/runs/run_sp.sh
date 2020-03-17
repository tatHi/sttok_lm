python manager.py --dataset $1 --segMode sp --clType lstm --useWord --gpuId $2 --spSize $3&
python manager.py --dataset $1 --segMode sp --clType lstm --useWord --sampling --gpuId $2 --spSize $3&
python manager.py --dataset $1 --segMode sp --clType lstm --useChar --gpuId $2 --spSize $3&

wait
python manager.py --dataset $1 --segMode sp --clType lstm --useChar --sampling --gpuId $2 --spSize $3&
python manager.py --dataset $1 --segMode sp --clType lstm --useChar --useWord --gpuId $2 --spSize $3&
python manager.py --dataset $1 --segMode sp --clType lstm --useChar --useWord --sampling --gpuId $2 --spSize $3&
