from chainer import Chain
from chainer import functions as F
from chainer import links as L
import compEmbed
import lookupEmbed
import cachedEmbed
class MultiEmbed(Chain):
    def __init__(self, charVocSize=None, charEmbedSize=None, wordVocSize=None, wordEmbedSize=None, useCache=False, gpuid=-1):
        super().__init__()

        assert wordEmbedSize is not None, 'wordEmbedSize must be specified'

        self.useComp = False
        self.useLookup = False
        self.useCache = useCache

        if charVocSize:
            assert charVocSize is not None, 'charVocSize must be given to set c2wEmbed'
            assert charEmbedSize is not None, 'charEmbedSize must be given to set c2wEmbed'
            comp = compEmbed.CompEmbed(charVocSize, charEmbedSize, wordEmbedSize)
            self.add_link('comp',comp)
            self.useComp = True
        if wordVocSize:
            if useCache:
                cache = cachedEmbed.CachedEmbed(wordVocSize, wordEmbedSize, gpuid) 
                self.add_link('cache',cache)
            else:    
                lookup = lookupEmbed.LookupEmbed(wordVocSize, wordEmbedSize)
                self.add_link('lookup',lookup)
            self.useLookup = True

        if charVocSize and wordVocSize:
            # use both embedding
            linear = L.Linear(2*wordEmbedSize, wordEmbedSize)
            self.add_link('linear',linear)

        print('set multiEmbed')
        print('comp/lookup/cache', self.useComp, self.useLookup, self.useCache)

    def forward(self, charIdLines=None, wordIdLines=None, charIdLines_cpu=None):
        #assert self.useComp==(charIdLines is not None), 'useComp? check inputs'
        #assert self.useLookup==(wordIdLines is not None), 'useLookup? check inputs'
        if self.useComp:
            ems_comp = self.comp(charIdLines)            
        if self.useLookup:
            if self.useCache:
                ems_lookup = self.cache(charIdLines_cpu, ems_comp)
            else:
                ems_lookup = self.lookup(wordIdLines)

        if self.useComp and not self.useLookup:
            return ems_comp
        elif not self.useComp and self.useLookup:
            return ems_lookup
        else:
            # sum
            #ems_sum = [c+l for c,l in zip(ems_comp, ems_lookup)]
            #return ems_sum

            # concat
            # 泥臭い方
            #ems = [self.linear(F.concat([c,l],axis=1)) for c,l in zip(ems_comp, ems_lookup)]
            
            # まとめてやる方
            #ls = [len(charIdLine) for charIdLine in charIdLines] if self.useCache else \
            #                [len(wordIdLine) for wordIdLine in wordIdLines]
            ls = [len(charIdLine) for charIdLine in charIdLines] 

            ems_comp = F.vstack(ems_comp)
            ems_lookup = F.vstack(ems_lookup)
            ems_flatten = self.linear(F.concat([ems_comp,ems_lookup],axis=1))
            
            ems = []
            flag = 0
            for l in ls:
                ems.append(ems_flatten[flag:flag+l])
                flag += l
            return ems

if __name__ == '__main__':
    import numpy as np
    charIdLines = [[np.array([1,2,3],'i'),np.array([2,],'i'),np.array([1,2,3,4],'i')],
                   [np.array([3,2],'i'),np.array([1,2,3],'i')]]
    wordIdLines = [np.array([2,3,4],'i'),np.array([1,2],'i')]

    me = MultiEmbed(charVocSize=10,charEmbedSize=2,wordVocSize=10,wordEmbedSize=3)

    print(me(charIdLines, wordIdLines))

