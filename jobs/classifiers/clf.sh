#!/bin/bash
#SBATCH --account=p31961

#SBATCH --partition=gengpu

#SBATCH --gres=gpu:a100:2

#SBATCH --ntasks-per-node=1

#SBATCH --mem=50G

#SBATCH --time=10:00:00

#SBATCH --job-name='lstmfcn_bayes'

JOB_NAME='lstmfcn_bayes'

curl -X POST -H 'Content-type: application/json' --data '{"text":"'${JOB_NAME}' started"}' $SLACK_WEBHOOK
source activate transphorm
python '/projects/p31961/transphorm/transphorm/experiments/aa_classifiers/aa_lstmfcn_bayes.py'

curl -X POST -H 'Content-type: application/json' --data '{"text":"'${JOB_NAME}' complete"}' $SLACK_WEBHOOK
