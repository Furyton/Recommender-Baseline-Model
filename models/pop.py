from abc import abstractmethod
import torch.nn as nn
import torch

from collections import Counter

from models.base import BaseModel

class POPModel(BaseModel):
    def __init__(self, args, dataset):
        super().__init__(args)

        item_train, item_valid, item_test, usernum, itemnum, rating_train, rating_valid, rating_test = dataset

        self.usernum = usernum
        self.itemnum = itemnum
        self.item_train = item_train
        self.item_valid = item_valid
        self.item_test = item_test

        self.T = args.T

        self.popularity = self.items_by_popularity()

        pop = [0] * (self.itemnum + 1)

        for i, v in self.popularity.items():
            pop[i] = v

        # item id starts at 1

        self.pop = torch.tensor(pop, dtype=torch.float, device=self.device, requires_grad=False)

        # self.pop_distr = torch.softmax(self.pop / self.T, 0).to(self.device)

        # print("[POPModel]: pop_distr dim: {}".format(self.pop_distr.size()))

    @classmethod
    def code(cls):
        return 'pop'

    def forward(self, batch):
        """
        input B x T
        output 
        """
        x = batch[0]
        batch_size = x.size()[0]
        length = x.size()[1]
        
        return self.pop.repeat(batch_size, length, 1)

    def items_by_popularity(self):
        popularity = Counter() 

        for user in range(0, self.usernum):
            popularity.update(self.item_train[user])
            popularity.update(self.item_valid[user])
            popularity.update(self.item_test[user])
        
        return popularity
    
    def predict(self, batch):
        seqs = batch[0]
        candidates = batch[1]
        scores= self.forward(seqs)  # B x T x V
        scores = scores[:, -1, :]  # B x V
        scores = scores.gather(1, candidates)  # B x C
        return scores
    
    def full_sort_predict(self, batch):
        seqs = batch[0]
        scores= self.forward(seqs)  # B x T x V
        scores = scores[:, -1, :]  # B x V
        return scores