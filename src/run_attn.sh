# dict
python manager.py --dataset $1 --sampling --useChar --useWord --segMode dict --clType attn --preEpoch mecab --gpuId 3 --outputPathAffix attn_unitrain_

# sentence piece
python manager.py --dataset $1 --sampling --useChar --useWord --segMode sp --clType attn --spSize 8000` --gpuId 3 --outputPathAffix attn_unitrain_
python manager.py --dataset $1 --sampling --useChar --useWord --segMode sp --clType attn --spSize 8000 --gpuId 3 --outputPathAffix attn_unitrain_



python manager.py --dataset $1 --sampling --useChar --useWord --segMode uni --clType attn --preEpoch 500 --gpuId 3 --outputPathAffix attn_unitrain_
# unitrain
python manager.py --dataset $1 --sampling --useChar --useWord --segMode uni --clType attn --preEpoch 500 --gpuId 3 --outputPathAffix attn_unitrain_
