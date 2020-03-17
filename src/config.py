maxLength = 8
charEmbedSize = 128
wordEmbedSize = 512
segHidSize = 512
clHidSize = 1024
classSize = 3

flexHyp = False
hyp = 1
alpha = hyp
beta = hyp
gamma = hyp

maxEpoch = 15

batchSize = 64

def setGPUID(gpuid):
    global xp
    gpuid = gpuid
    if gpuid >= 0:
        from chainer import cuda
        xp = cuda.cupy
        cuda.get_device(gpuid).use()
    else:
        import numpy
        xp = numpy

xp = None
gpuid = None

reduceAfterSampling = False
