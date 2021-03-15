from .base import BaseModel
from .bert_modules.bert import BERT
import torch.nn as nn


class BERTModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.bert = BERT(args)
        self.out = nn.Linear(self.bert.hidden, args.bert_num_items + 1)
        self.is_original = args.original
    @classmethod
    def code(cls):
        return 'bert'

    def forward(self, x, test):
        if self.is_original:
            return self.out(self.bert(x, test))
        else:
            if not test:
                x, y = self.bert(x, test)
            else:
                x = self.bert(x, test)
            return (self.out(x), self.out(y)) if not test else self.out(x)
