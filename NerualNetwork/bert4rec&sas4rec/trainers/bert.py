from .base import AbstractTrainer
from .utils import recalls_ndcgs_and_mrr_for_ks

import torch.nn as nn
import torch


class BERTTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        self.ce = nn.CrossEntropyLoss(ignore_index=0)

    @classmethod
    def code(cls):
        return 'bert'

    def add_extra_loggers(self):
        with torch.no_grad():
            dataiter = iter(self.train_loader)
            seqs, labels = dataiter.next()

            self.writer.add_graph(self.model, seqs)

    def log_extra_train_info(self, log_data):
        pass

    def close_training(self):
        pass

    def calculate_loss(self, batch):
        batch = [x.to(self.device) for x in batch]

        seqs, labels = batch
        logits = self.model(seqs)  # B x T x V

        logits = logits.view(-1, logits.size(-1))  # (B*T) x V

        labels = labels.view(-1)  # B*T

        loss = self.ce(logits, labels)
        return loss

    def calculate_metrics(self, batch):
        batch = [x.to(self.device) for x in batch]
        
        seqs, candidates, labels = batch
        scores = self.model(seqs)  # B x T x V
        scores = scores[:, -1, :]  # B x V
        scores = scores.gather(1, candidates)  # B x C

        metrics = recalls_ndcgs_and_mrr_for_ks(scores, labels, self.metric_ks)
        return metrics
