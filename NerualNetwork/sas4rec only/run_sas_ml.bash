#!/bin/bash
#SBATCH -e sas_ml.err
#SBATCH -o sas_ml.out
#SBATCH -J sas_ml

#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=999:00:00
#SBATCH --mem=50G

python  main.py


