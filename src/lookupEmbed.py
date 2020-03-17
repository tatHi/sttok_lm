from chainer import Chain
from chainer import links as L
class LookupEmbed(Chain):
    def __init__(self,vocSize,embedSize):
        super().__init__(embed = L.EmbedID(vocSize, embedSize))
    
    def forward(self, idLines):
        return [self.embed(idLine) for idLine in idLines]
