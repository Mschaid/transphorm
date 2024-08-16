#!/bin/bash
#SBATCH --account=p31961

#SBATCH --partition=gengpu

#SBATCH --gres=gpu:a100:2

#SBATCH --ntasks-per-node=1

#SBATCH --mem=50G

#SBATCH --time=2:00:00

#SBATCH --job-name='inference'

JOB_NAME='inference'

curl -X POST -H 'Content-type: application/json' --data '{"text":"'${JOB_NAME}' started"}' $SLACK_WEBHOOK
source activate transphorm
python '/projects/p31961/transphorm/transphorm/experiments/aa_classifiers/inference.py'

curl -X POST -H 'Content-type: application/json' --data '{"text":"'${JOB_NAME}' complete"}' $SLACK_WEBHOOK
