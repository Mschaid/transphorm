#!/bin/bash

#SBATCH --account=p31961

#SBATCH --partition=normal

#SBATCH --nodes=10

#SBATCH --qos=normal
#SBATCH --ntasks-per-node=1

#SBATCH --mem=200G

#SBATCH --time=48:00:00

#SBATCH --job-name='rocket_trainer'
JOB_NAME='rocket_bayes_tuning'

curl -X POST -H 'Content-type: application/json' --data '{"text":"'${JOB_NAME}' started"}' $SLACK_WEBHOOK
source activate transphorm
python '/projects/p31961/transphorm/transphorm/experiments/rocket/rocket_bayes_tuning.py'

curl -X POST -H 'Content-type: application/json' --data '{"text":"'${JOB_NAME}' complete"}' $SLACK_WEBHOOK
