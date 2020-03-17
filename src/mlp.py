from chainer import Chain
from chainer import links as L
from chainer import functions as F

activations= {
        'tanh': F.tanh,
        'relu': F.relu,
        'sigmoid': F.sigmoid
    }

class MLP(Chain):
    def __init__(self, sizes, activation='sigmoid', noActFinal=False):
        super().__init__()

        self.numOfLayer = len(sizes)-1
        for i in range(self.numOfLayer):
            inSize, outSize = sizes[i:i+2]
            linear = L.Linear(inSize, outSize)
            self.add_link('linear%d'%i, linear)
      
        self.act = activations[activation]
        self.noActFinal = noActFinal

    def forward(self, x):
        y = x
        for i,linear in enumerate(self.links()):
            if i==0:
                # the first object of self.links is MLP object itself.
                continue

            y = linear(y)
            if i==self.numOfLayer and self.noActFinal:
                pass
            else:
                y = self.act(y)
        return y
        
if __name__=='__main__':
    m = MLP([3,2,5], 'tanh', True)
    import numpy as np
    x = np.array([[1,2,3],[2,3,4]], 'f')
    print(m.forward(x))
