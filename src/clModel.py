from chainer import Chain
from chainer import functions as F
from chainer import links as L
import chainer

class CLModel(Chain):
    def __init__(self):
        super().__init__()

    def __call__(self):
        return None

    def getLoss(self, ws, labels):
        ys = self.__call__(ws)
        loss = F.softmax_cross_entropy(ys, labels)
        return loss

class LSTMModel(CLModel):
    def __init__(self, inSize, hidSize, outSize):
        # out size is usually class size
        super().__init__()

        lstm = L.NStepLSTM(1,inSize, hidSize, dropout=0.5)
        linear = L.Linear(hidSize, outSize)

        self.add_link('bilstm',lstm)
        self.add_link('linear',linear)
        
    def __call__(self, ws):
        ws = [F.dropout(w, 0.5) for w in ws] # embed dropout
        _,hs,_ = self.bilstm(None,None,ws)
        hs = F.reshape(hs,(len(ws),-1))
        ys = self.linear(F.dropout(hs,0.5))
        return ys

class BiLSTMModel(CLModel):
    def __init__(self, inSize, hidSize, outSize):
        # out size is usually class size
        super().__init__()

        bilstm = L.NStepBiLSTM(1,inSize, hidSize, dropout=0.3)
        linear = L.Linear(hidSize*2, outSize)

        self.add_link('bilstm',bilstm)
        self.add_link('linear',linear)
        
    def __call__(self, ws):
        hy,_,_ = self.bilstm(None,None,ws)
        hy = F.hstack(hy)
        ys = self.linear(hy)
        return ys

class DANModel(CLModel):
    def __init__(self, inSize, hidSize, outSize):
        super().__init__()

        linear1 = L.Linear(inSize, hidSize)
        linear2 = L.Linear(hidSize, outSize)

        self.add_link('linear1', linear1)
        self.add_link('linear2', linear2)

    def __call__(self, ws):
        hs = [F.average(w,axis=0) for w in ws]
        hs = F.stack(hs)
        xs = F.tanh(self.linear1(hs))
        ys = F.tanh(self.linear2(xs))
        return ys

import attentionEncoder
import mlp as M
class AttentiveModel(CLModel):
    def __init__(self, inSize, outSize):
        super().__init__()
        attn = attentionEncoder.AttentionEncoder(
            inputSize = inSize,
            outputSize = 1024,
            attentionHidSize = 512,
            attentionRowSize = 3)
        mlp = M.MLP([1024, 256, 512, outSize], 'tanh', noActFinal=True)
        
        self.add_link('attn', attn)
        self.add_link('mlp', mlp)

    def __call__(self, ws):
        xs, attnLoss = self.attn(ws)
        ys = self.mlp(F.dropout(xs, 0.5))
        
        if chainer.config.train:        
            return ys, attnLoss 
        else:
            return ys

    def getLoss(self, ws, labels):
        ys, attnLoss = self.__call__(ws)
        labelLoss = F.softmax_cross_entropy(ys, labels)
        return labelLoss + attnLoss

if __name__ == '__main__':
    import numpy as np
    ws = [np.random.rand(5,2).astype('f'),np.random.rand(3,2).astype('f')]
    ts = np.array([0,2],'i')

    am = AttentiveModel(2,3)
    print(am.getLoss(ws,ts))

