from abc import *
from config import *

import torch

class AbstractBaseTrainer(metaclass=ABCMeta):
    def __init__(self, args, model, export_root):
        self.args = args
        self.device = args.device
        self.model = model.to(self.device)

        self.is_parallel = args.num_gpu > 1
        if self.is_parallel:
            self.model = torch.nn.DataParallel(self.model)

        self.export_root = export_root
        
        self.batch_size = args.train_batch_size

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def _load_state(self):
        pass

    @abstractmethod
    def train(self):
        pass
    
    def _create_state_dict(self):
        return {
            STATE_DICT_KEY: self.model.module.state_dict() if self.is_parallel else self.model.state_dict(),
            OPTIMIZER_STATE_DICT_KEY: self.optimizer.state_dict(),
        }