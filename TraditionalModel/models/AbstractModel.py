from abc import *


class BaseModel(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @classmethod
    @abstractmethod
    def fit(cls):
        pass

    @abstractmethod
    def predict(self):
        pass
