from chainer import Chain

class Model(Chain):
    def __init__(self, embedModel, clModel):
        super().__init__()
        self.add_link('embedModel',embedModel)
        self.add_link('clModel',clModel)

    def forward(self, char_ids, word_ids, char_ids_cpu):
        ems = self.embedModel(char_ids, word_ids, char_ids_cpu)
        ys = self.clModel(ems)        
        return ys

    def getLoss(self, char_ids, word_ids, labels, char_ids_cpu):
        ems = self.embedModel(char_ids,word_ids,char_ids_cpu)
        loss = self.clModel.getLoss(ems,labels) 
        return loss

