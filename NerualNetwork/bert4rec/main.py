from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import *
import json


def train():
    export_root = setup_train(args)

    train_loader, val_loader, test_loader = dataloader_factory(args)
    model = model_factory(args)


"""
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    #trainer.train()
    # test_result = test_with(trainer.best_model, test_loader)
    # save_test_result(export_root, test_result)
    trainer.test()
"""


def test():
    pass

if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
    else:
        pass
