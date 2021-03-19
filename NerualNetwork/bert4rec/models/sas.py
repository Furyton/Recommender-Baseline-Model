from .base import BaseModel
import torch.nn as nn
from .sas_model import sas


class SASModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.sas = SASModel(args)

    @classmethod
    def code(cls):
        return 'sas'

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):  # for training
        return self.sas(user_ids, log_seqs, pos_seqs, neg_seqs)

    def predict(self, user_ids, log_seqs, item_indices):  # for inference
        return self.sas.predict(user_ids, log_seqs, item_indices)
