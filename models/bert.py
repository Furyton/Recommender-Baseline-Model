from .base import BaseModel
from .bert_modules.bert import BERT
import torch.nn as nn


class BERTModel(BaseModel):
    def __init__(self, args, dataset):
        super().__init__(args)
        self.bert = BERT(args)
        self.out = nn.Linear(self.bert.hidden, args.num_items + 1)
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
        self.sample_wise_ce = nn.CrossEntropyLoss(reduction='none', ignore_index=0)

    @classmethod
    def code(cls):
        return 'bert'

    def forward(self, batch):
        # seqs, labels, rating, seq_lens, user = batch
        seqs = batch[0]
        return self.out(self.bert(seqs))

    def predict(self, batch):
        # seqs, candidates, labels = batch
        candidates = batch[1]
        scores = self.forward(batch)  # B x T x V
        scores = scores[:, -1, :]  # B x V
        scores = scores.gather(1, candidates)  # B x C
        return scores

    def full_sort_predict(self, batch):
        # seqs, labels = batch
        scores = self.forward(batch)  # B x T x V
        # print("[bert]: scores size" + scores.size())
        scores = scores[:, -1, :].squeeze()  # B x V
        # print("[bert]: after scores size" + scores.size())
        return scores