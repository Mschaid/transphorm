#!/bin/bash
#SBATCH -A p31961
#SBATCH -p gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 1 
#SBATCH --mem=10G
source activate transphorm
python test_rocket.py
