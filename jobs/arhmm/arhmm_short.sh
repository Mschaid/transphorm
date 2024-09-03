#!/bin/bash
#SBATCH --account=p31961

#SBATCH --partition=short

#SBATCH --ntasks-per-node=1

#SBATCH --mem=100G

#SBATCH --time=04:00:00

#SBATCH --job-name='arhmm_short'

#SBATCH --output='arhmm.log'

JOB_NAME='arhmm'

curl -X POST -H 'Content-type: application/json' --data '{"text":"'${JOB_NAME}' started"}' $SLACK_WEBHOOK
source activate transphorm
python '/projects/p31961/transphorm/transphorm/experiments/arhmm/arhmm_experiment.py'



curl -X POST -H 'Content-type: application/json' --data '{"text":"'${JOB_NAME}' complete"}' $SLACK_WEBHOOK
