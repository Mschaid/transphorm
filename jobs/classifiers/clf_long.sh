#!/bin/bash
#SBATCH --account=p31961

#SBATCH --partition=gengpu

#SBATCH --gres=gpu:a100:4

#SBATCH --ntasks-per-node=1

#SBATCH --mem=50G

#SBATCH --time=12:00:00

#SBATCH --job-name='lstmnfcn_bayes_tuning_5_day'

JOB_NAME='lstmnfcn_bayes_tuning_5_day'

curl -X POST -H 'Content-type: application/json' --data '{"text":"'${JOB_NAME}' started"}' $SLACK_WEBHOOK
source activate transphorm
python '/projects/p31961/transphorm/transphorm/experiments/aa_classifiers/aa_lstmfcn_bayes_5_day.py'

curl -X POST -H 'Content-type: application/json' --data '{"text":"'${JOB_NAME}' complete"}' $SLACK_WEBHOOK
