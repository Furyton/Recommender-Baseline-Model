#!/bin/bash
#SBATCH -e bert_ml.err
#SBATCH -o bert_ml.out
#SBATCH -J bert_ml

#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --mem=70G

python  main.py


