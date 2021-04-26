from models import MODELS

import argparse

parser = argparse.ArgumentParser(description='Traditional model')


parser.add_argument('--config_file', type=str, default='config.json')
parser.add_argument('--dataset_name', type=str)
parser.add_argument('--model_code', type=str, choices=MODELS.keys())

# BPR
parser.add_argument('--dim', type=int, default=64, help='latent dimension')
parser.add_argument('--iteration', type=int, default=200, help='number of epoch for training')
parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
parser.add_argument('--regularization', type=float, default=0.0, help='regularization for user and item embedding')

args = parser.parse_args()
