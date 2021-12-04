import torch.nn as nn
from abc import *


class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def predict(self, seqs, candidates, labels):
        pass

    @abstractmethod
    def full_sort_predict(self, seqs, labels):
        pass

