from .POP import POP_model
from .ItemKNN import itemKNN_model
from .BPR_MF import BPR_MF_model

MODELS = {
    POP_model.code(): POP_model,
    itemKNN_model.code(): itemKNN_model,
    BPR_MF_model.code(): BPR_MF_model
}


def get_model(code, args, train, test, candidates, item_num, user_num):
    model = MODELS[code]

    return model(args, train, test, candidates, item_num, user_num)
