# Stochastic Tokenization with a Language Model for Neural Text Classification
WIP

# requirements
- Chainer 5.0 <
- Cython

# setup
```
$ cd src
$ sh compile.sh
```

# make initial language model with unsupervised learning
If you want to use language model initialized with simple unsupervised word segmentation.

```
$ python unigramTrain.py -i ../data/toy_train_text.txt \
                         -o ../result/toy.lm \
                         -e 100
```
Note that the option `-e` or `--epoch` should be more than 100 because the unsupervised training with Gibbs sampling requires many updating.

# run trainer with updating the language model
Running `main.py` with following options trains the neural text classifier and the language model.
Results of text classification at the end of each epochs and trained models are dumped to the specified path.

```
$ mkdir ../result/res_toy
$ python main.py --trainText ../data/toy_train_text.txt \ 
                 --trainLabel ../data/toy_train_label.txt \
                 --validText ../data/toy_valid_text.txt \
                 --validLabel ../data/toy_valid_label.txt \
                 --testText ../data/toy_test_text.txt \
                 --testLabel ../data/toy_test_label.txt \
                 --dumpTo ../result/ \
                 --classSize 2 \
                 --sampling \
                 --useChar \
                 --useWord \
                 --useCache \
                 --uniTrain \
                 --segMode uni \
                 --clType dan \
                 --smoothingHyp 1 \
                 --gpuId 0
```
