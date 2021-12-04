from models.GRU4rec import GRU4RecModel
from models.bert import BERTModel
from models.pop import POPModel
from models.Caser import CaserModel

MODELS = {
    BERTModel.code(): BERTModel,
    POPModel.code(): POPModel,
    GRU4RecModel.code(): GRU4RecModel,
    CaserModel.code(): CaserModel,
}


def model_factory(args, model_type: str, dataset: list):
    if model_type.lower() not in MODELS.keys():
        raise NotImplementedError

    model = MODELS[model_type]

    return model(args, dataset)
