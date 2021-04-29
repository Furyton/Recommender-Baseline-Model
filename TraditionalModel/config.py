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

# KNN

parser.add_argument('--k', type=int, default=100, help='maximum similar item number')
parser.add_argument('--lmbd', type=float, default=20, help='Regularization. Discounts the similarity of rare items (incidental co-occurrences). (Default value: 20)')
parser.add_argument('--alpha', type=float, default=0.5, help='Balance between normalizing with the supports of the two items. 0.5 gives cosine similarity, 1.0 gives confidence (as in association rules).')
# lmbd : float
#         Regularization. Discounts the similarity of rare items (incidental co-occurrences). (Default value: 20)
#     alpha : float
#         Balance between normalizing with the supports of the two items. 0.5 gives cosine similarity, 1.0 gives confidence (as in association rules).
#


args = parser.parse_args()
