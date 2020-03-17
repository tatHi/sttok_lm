# sh run_uni_ch.sh hotel_ch dan 2 8000 500 jieba 5
#python manager.py --dataset $1 --segMode uni --clType $2 --useChar --sampling --gpuId $3 --preEpoch $5 --outputPathAffix $5\_ --smoothingHyp $7&
#python manager.py --dataset $1 --segMode uni --clType $2 --useChar --sampling --gpuId $3 --preEpoch $6 --outputPathAffix $6\_ --smoothingHyp $7&
#wait
python manager.py --dataset $1 --segMode uni --clType $2 --useChar --uniTrain --gpuId $3 --preEpoch $6 --outputPathAffix $6\_ --smoothingHyp $7&
python manager.py --dataset $1 --segMode uni --clType $2 --useChar --uniTrain --gpuId $3 --preEpoch $5 --outputPathAffix $5\_ --smoothingHyp $7&
wait
