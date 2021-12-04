'''
    all the options for dataset, dataloader, model and trainer
'''

from config import *
from models import MODELS
from dataloaders import DATALOADERS

import argparse

parser = argparse.ArgumentParser(description='RecPlay')

################
# Top Level
################
parser.add_argument('--config_file', type=str, default='config.json')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'resume'])
################
# Test
################
parser.add_argument('--test_model_path', type=str, default=None)

parser.add_argument('--model_state_path', type=str, default=None)

################
# Dataset
################
parser.add_argument('--load_processed_dataset', type=bool, default=False)
parser.add_argument('--save_processed_dataset', type=bool, default=True)
parser.add_argument('--dataset_cache_filename', type=str)
parser.add_argument('--split', type=str, default='leave_one_out', help='How to split the datasets')
parser.add_argument('--min_length', type=int, default=3, help='minimum length for each user')
parser.add_argument('--min_item_inter', type=int, default=5, help='minimum interaction for each item')
parser.add_argument('--good_only', type=bool, default=True, help='only use items user likes')
parser.add_argument('--do_reindex', type=bool, default=True)
parser.add_argument('--use_rating', type=bool, default=True)
################
# Dataloader
################
parser.add_argument('--dataloader_type', type=str, default='mask', choices=DATALOADERS.keys())
parser.add_argument('--dataloader_random_seed', type=float, default=0.0)
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=64)
parser.add_argument('--prop_sliding_window', type=float, default=0.1)
parser.add_argument('--worker_number', type=int, default=1)
################
# NegativeSampler
################
parser.add_argument('--train_negative_sampler_code', type=str, default='random', choices=['popular', 'random'],
                    help='Method to sample negative items for training. Not used in bert')
parser.add_argument('--train_negative_sample_size', type=int, default=100)
parser.add_argument('--train_negative_sampling_seed', type=int, default=None)
parser.add_argument('--test_negative_sampler_code', type=str, default='random', choices=['popular', 'random'],
                    help='Method to sample negative items for evaluation')
parser.add_argument('--test_negative_sample_size', type=int, default=100)
parser.add_argument('--test_negative_sampling_seed', type=int, default=None)

################
# Trainer
################
# device #
parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
parser.add_argument('--num_gpu', type=int, default=1)
parser.add_argument('--device_idx', type=str, default='0') # [0, 1, 2 ... ]
# optimizer #
parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'])
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='l2 regularization')
parser.add_argument('--momentum', type=float, default=None, help='SGD momentum')
# lr scheduler #
parser.add_argument('--decay_step', type=int, default=15, help='Decay step for StepLR')
parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for StepLR')
# epochs #
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training')
# logger #
parser.add_argument('--log_period_as_iter', type=int, default=12800)
# evaluation #
parser.add_argument('--metric_ks', nargs='+', type=int, default=[10, 20, 50], help='ks for Metric@k')
parser.add_argument('--best_metric', type=str, default='NDCG@10', help='Metric for determining the best model')
# processing #
parser.add_argument('--show_process_bar', type=bool, default=False, help='show the processing bar or not')
################
# Model
################
parser.add_argument('--enable_mentor', type=bool)
parser.add_argument('--mentor_model', type=str, default='pop', choices=MODELS.keys())

parser.add_argument('--enable_sample', type=bool, default=False)
parser.add_argument('--samples_ratio', type=float, default=0.5)

parser.add_argument('--model_code', type=str, default='bert', choices=MODELS.keys())
parser.add_argument('--model_init_seed', type=int, default=None)

parser.add_argument('--max_len', type=int, default=50, help='Length of sequence')
parser.add_argument('--embed_size', type=int, default=64, help='Embedding size')
parser.add_argument('--hidden_units',type=int, default=64, help='Size of hidden vectors (d_model)')
parser.add_argument('--num_heads', type=int, default=2, help='Number of heads for multi-attention')
parser.add_argument('--hidden_dropout', type=float, default=0.2)
parser.add_argument('--attention_dropout',type=float, default=0.2)

parser.add_argument('--training_stage', type=str, default=NORMAL_STAGE, choices=[PRETRAIN_STAGE, FINE_TUNE_STAGE, NORMAL_STAGE])

# BERT #
parser.add_argument('--bert_num_blocks', type=int, default=None, help='Number of transformer layers')
parser.add_argument('--bert_mask_prob', type=float, default=None, help='Probability for masking items in the training sequence')

# GRU4Rec #
parser.add_argument('--gru_num_layers', type=int, default=2, help='Number of GRU layers')
parser.add_argument('--caser_hidden_dropout', type=float, default=0.5, help='dropout rate in CaserModel')

# Caser #
parser.add_argument('--caser_nh', type=int, default=8, help='Caser nh')
parser.add_argument('--caser_nv', type=int, default=16, help='Caser nv')

################
# Experiment
################
parser.add_argument('--experiment_dir', type=str, default='experiments')
parser.add_argument('--experiment_description', type=str, default='test')
parser.add_argument('--dataset_name', type=str, default=None)
parser.add_argument('--subdataset_rate', type=float, default=0.1)

parser.add_argument('--validation_rate', type=float, default=0.2)

parser.add_argument('--num_items', type=int, default=None, help='Number of total items')
parser.add_argument('--start_index', type=int, default=1)

# if using slurm

parser.add_argument('--slurm_log_file_path', type=str, default=None)

################
args = parser.parse_args()

