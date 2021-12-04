#!/bin/bash

#SBATCH --partition=debug

#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00

#SBATCH --mem=16G

python main.py --slurm_log_file_path=${SAVED_PATH}/${JOB_NAME}.${SLURM_JOB_ID}.out --config_file=$CONFIG_PATH

