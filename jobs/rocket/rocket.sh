#!/bin/bash

#SBATCH --account=p31961

#SBATCH --partition=normal

#SBATCH --nodes=1

#SBATCH --qos=normal
#SBATCH --ntasks-per-node=4

#SBATCH --mem=10G

#SBATCH --time=00:10:00

#SBATCH --job-name='rocket_trainer'

JOB_NAME='rocket_trainer'

curl -X POST -H 'Content-type: application/json' --data '{"text":"'${JOB_NAME}' started"}' $SLACK_WEBHOOK
python '/projects/p31961/transphorm/transphorm/experiments/rocket/rocket.py'

curl -X POST -H 'Content-type: application/json' --data '{"text":"'${JOB_NAME}' complete"}' $SLACK_WEBHOOK