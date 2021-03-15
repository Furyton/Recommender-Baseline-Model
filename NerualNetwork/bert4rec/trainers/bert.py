import torch

from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks

import torch.nn as nn


class BERTTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
        self.is_original = args.original

    @classmethod
    def code(cls):
        return 'bert'

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def calculate_loss(self, batch):
        if self.is_original == 1:
            seqs, labels = batch
            logits = self.model(seqs, False)  # B x T x V

            logits = logits.view(-1, logits.size(-1))  # (B*T) x V

            labels = labels.view(-1)  # B*T

            loss = self.ce(logits, labels)
            return loss
        else:
            seqs, labels = batch
            logits, logits_2 = self.model(seqs, False)  # B x T x V

            logits = logits.view(-1, logits.size(-1))  # (B*T) x V
            logits_2 = logits_2.view(-1, logits_2.size(-1))

            labels_2 = torch.where(labels == 0, seqs, labels)

            for u in range(len(labels_2)):
                for i in range(len(labels_2[u])):
                    if labels_2[u][i] != 0:
                        labels_2[u][i] = 0
                        break

            labels_2 = labels_2.view(-1)

            labels = labels.view(-1)  # B*T

            loss = self.ce(logits, labels) + self.ce(logits_2, labels_2)
            return loss

    def calculate_metrics(self, batch):
        seqs, candidates, labels = batch
        scores= self.model(seqs, True)  # B x T x V
        scores = scores[:, -1, :]  # B x V
        scores = scores.gather(1, candidates)  # B x C

        metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        return metrics
