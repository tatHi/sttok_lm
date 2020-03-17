#gold
python manager.py --dataset news_ja --useChar --useWord --segMode dict --clType lstm --preEpoch mecab --spSize 8000 --gpuId $1 --outputPathAffix dict_
#subword
python manager.py --dataset news_ja --useChar --useWord --segMode sp --clType lstm --spSize 8000 --gpuId $1 --outputPathAffix subword_
#subword+smp
python manager.py --dataset news_ja --useChar --useWord --segMode sp --sampling --clType lstm --spSize 8000 --gpuId $1 --outputPathAffix subwordsmp_
#proposed+sp
python manager.py --dataset news_ja --useChar --useWord --segMode uni --sampling --uniTrain --clType lstm --preEpoch mecab --spSize 8000 --gpuId $1 --outputPathAffix mecab_
#proposed+unsp
python manager.py --dataset news_ja --useChar --useWord --segMode uni --sampling --uniTrain --clType lstm --preEpoch 500 --spSize 8000 --gpuId $1 --outputPathAffix 500_
