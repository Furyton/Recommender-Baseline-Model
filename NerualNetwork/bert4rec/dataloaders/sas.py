from .base import AbstractDataloader

import torch
import torch.utils.data as data_utils
import copy

class SASDataLoader(AbstractDataloader):
    def __init__(self, args, dataset):
        super(self).__init__(args, dataset)
        self.max_len = args.sas_max_len
