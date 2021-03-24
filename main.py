import torch

from options import args
import models
import dataloaders
import trainers
import utils


def train():
    export_root = utils.setup_train(args)
    train_loader, val_loader, test_loader = dataloaders.factory(args)
    model = models.factory(args)
    trainer = trainers.factory(args, model, train_loader, val_loader, test_loader, export_root)
    trainer.train()

    test_model = (input('Test model with test dataset? y/[n]: ') == 'y')
    if test_model:
        trainer.test()


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    else:
        raise ValueError('Invalid mode')
