from chainer import Chain
from chainer import functions as F
from chainer import links as L
import chainer
from copy import copy

class SelfAttention(Chain):
    def __init__(self, inputSize, attentionHidSize, attentionRowSize):
        super().__init__()
        
        linear1 = L.Linear(inputSize, attentionHidSize, nobias=True)
        linear2 = L.Linear(attentionHidSize, attentionRowSize, nobias=True)

        self.add_link('linear1',linear1)
        self.add_link('linear2',linear2)

        # to utilize attention loss, identity matrix is required
        # initialize this variable when calcAttentionLoss method is called at first. 
        # because this class cannot receive xp
        self.I = None

    def forward(self, xs):
        A = [self.calcAttention(x) for x in xs]

        # sentence embedding vectors M
        M = F.vstack(
            [F.flatten(
                F.matmul(
                    A[i].T, xs[i]
                )
            ) for i in range(len(xs))]
        )
       
        
        if chainer.config.train:
            loss = self.calcAttentionLoss(A)
        else:
            loss = None
        
        return M, loss

    def calcAttention(self, y):
        h = F.tanh(self.linear1(y))
        a = F.softmax(self.linear2(h))
        return a

    def calcAttentionLoss(self, A):
        if self.I is None:
            self.setIdentityMatrix(A)

        # rewrite to use comprehension like
        # F.sum([... for a in A])
        '''
        loss = 0
        for a in A:
            # |AA^T|_F^2 = |B|_F^2 = tr(BB^T) = sum(BB^T*I)
            b = F.matmul(a.T, a)-self.I
            loss += F.sum(F.diagonal(F.matmul(b,b.T)))
        '''
        losses = []
        for a in A:
            # |AA^T|_F^2 = |B|_F^2 = tr(BB^T) = sum(BB^T*I)
            b = F.matmul(a.T, a)-self.I
            losses.append(F.sum(F.diagonal(F.matmul(b,b.T))))
        return F.average(F.vstack(losses))

    def setIdentityMatrix(self, A):
        # take arg given to calcAttentionLoss
        a = A[0]
        b = F.matmul(a.T, a)
        self.I = copy(b.data)
        chainer.initializers.Identity(1)(self.I)

if __name__=='__main__':
    import numpy as np

    bilstm = L.NStepBiLSTM(1,2,3,0.0)
    attention = SelfAttention(6,2,3)

    xs = [np.random.random((2,2)).astype('f') for i in range(3)]
    _, _, ys = bilstm(None, None, xs)
    print(attention.forward(ys))
