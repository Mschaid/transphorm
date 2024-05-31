#!/bin/bash
#SBATCH --account=p31961

#SBATCH --partition=normal

#SBATCH --nodes=12

#SBATCH --ntasks-per-node=12

#SBATCH --mem=10G

#SBATCH --time=12:00:00

#SBATCH --job-name='rocket_trainer'

#SBATCH --output=rocket_trainer.out


module purge
module load python-anaconda3
source activate transphorm

python '/projects/p31961/transphorm/transphorm/experiments/rocket/rocket.py' 