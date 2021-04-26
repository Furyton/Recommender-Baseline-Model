import numpy as np
from .AbstractModel import BaseModel


class POP_model(BaseModel):
    def __init__(self):
        super().__init__()

    @classmethod
    def code(cls):
        return 'pop'

    def fit(self):
        pass

    def predict_next(self):
        pass
