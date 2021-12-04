from abc import *
from numpy import random


# [user_train, user_valid, user_test, usernum, itemnum]


class AbstractDataloader(metaclass=ABCMeta):
    def __init__(self, args, dataset):
        self.args = args
        self.seed = args.dataloader_random_seed
        # TODO
        self.rng = random  # rng.random(l, r) -> rand int \in [l, r - 1]
        self.save_folder = ''
        self.worker_num = args.worker_number


    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def get_pytorch_dataloaders(self):
        pass
