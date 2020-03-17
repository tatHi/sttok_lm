from chainer import Chain
from chainer import functions as F
from chainer import links as L
import chainer
import selfAttention

class AttentionEncoder(Chain):
    def __init__(self, **kwargs):
        super().__init__()

        inputSize = kwargs['inputSize']
        outputSize = kwargs['outputSize']
        attentionHidSize = kwargs['attentionHidSize']
        attentionRowSize = kwargs['attentionRowSize']
        
        assert outputSize%2==0, 'outputSize must be debide by 2. half of num is assigned to each directional lstm'

        bilstm = L.NStepBiLSTM(1, inputSize, outputSize//2, 0.25)
        attn = selfAttention.SelfAttention(inputSize=outputSize, 
                                           attentionHidSize=attentionHidSize, 
                                           attentionRowSize=attentionRowSize)
        linear = L.Linear(outputSize*attentionRowSize, outputSize)

        self.add_link('bilstm', bilstm)
        self.add_link('attn', attn)
        self.add_link('linear', linear)

        #self.I = xp.eye(attentionRowSize).astype('f')

    def forward(self, xs):
        hy, cy, ys = self.bilstm(hx=None, cx=None, xs=xs)
        ys, loss = self.attn(ys)       
        ys = self.linear(ys)
        return ys, loss

if __name__=='__main__':
    import numpy as np
    inputSize = 3
    outputSize =5
    xs = [np.random.random((5, inputSize)).astype('f') for i in range(3)]

    ae = AttentionEncoder(inputSize=inputSize, 
                          attentionHidSize=5,
                          attentionRowSize=3,
                          outputSize=outputSize,
                          xp=np)

    o = ae.forward(xs)
    print(o.shape)
