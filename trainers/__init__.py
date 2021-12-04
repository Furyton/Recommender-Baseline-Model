from config import FINE_TUNE_STAGE, NORMAL_STAGE, PRETRAIN_STAGE
from trainers.NormalTrainer import NormalTrainer
from trainers.SoftRecTrainer import SoftRecTrainer

TRAINERS = {
    FINE_TUNE_STAGE: SoftRecTrainer,
    NORMAL_STAGE: NormalTrainer,
    PRETRAIN_STAGE: NormalTrainer
}


def trainer_factory(args, model, train_loader, val_loader, test_loader, dataset, export_root, mode:str, mentor=None):
    trainer = TRAINERS[mode]
    # return trainer(args, model, train_loader, val_loader, test_loader, export_root)

    if mode.lower() == PRETRAIN_STAGE:
        return trainer(args, model, train_loader, val_loader, test_loader, export_root, 'mentor_models')
    elif mode.lower() == FINE_TUNE_STAGE:
        return trainer(args, model, train_loader, val_loader, test_loader, export_root, mentor)
    elif mode.lower() == NORMAL_STAGE:
        return trainer(args, model, train_loader, val_loader, test_loader, export_root, 'models')
    else:
        raise ValueError

    # if mode.lower() == 'student':
    #     return trainer(args, model, train_loader, export_root)
    # elif mode.lower() == 'mentor':
    #     return trainer(args, model, train_loader, val_loader, test_loader, export_root)
    # elif mode.lower() == 'curriculum':
    #     return trainer(args, model, train_loader, val_loader, test_loader, export_root, mentor)
    # elif mode.lower() == 'normal':
    #     return trainer(args, model, train_loader, val_loader, test_loader, export_root)
    # else:
    #     raise ValueError
