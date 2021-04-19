from .base import AbstractTrainer
from .utils import recalls_ndcgs_and_mrr_for_ks
import torch.nn as nn
import torch
import numpy as np


class SASTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        self.bce_criterion = nn.BCEWithLogitsLoss()

    @classmethod
    def code(cls):
        return 'sas'

    def add_extra_loggers(self):
        # with torch.no_grad():
        #     dataiter = iter(self.train_loader)
        #     seqs, labels = dataiter.next()
        #
        #     self.writer.add_graph(self.model, seqs)
        pass

    def log_extra_train_info(self, log_data):
        pass

    def calculate_loss(self, batch):
        seq, pos, neg = batch
        pos_logits, neg_logits = self.model(seq, pos, neg)
        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.args.device), torch.zeros(neg_logits.shape, device=self.args.device)

        indices = torch.where(pos.cpu() != 0)

        loss = self.bce_criterion(pos_logits[indices], pos_labels[indices])
        loss += self.bce_criterion(neg_logits[indices], neg_labels[indices])

        return loss

    def calculate_metrics(self, batch):
        seqs, candidates, labels = batch
        logits = self.model.predict(seqs, candidates)

        metrics = recalls_ndcgs_and_mrr_for_ks(logits, labels, self.metric_ks)

        return metrics
