#!/bin/bash
#SBATCH --account=p31961

#SBATCH --partition=normal

#SBATCH --ntasks-per-node=1

#SBATCH --mem=150G

#SBATCH --time=32:00:00

#SBATCH --job-name='arhmm_full_learning_partitioned'

#SBATCH --output='arhmm_full_learning_partitioned.log'

JOB_NAME='arhmm_full_learning_partitioned'

curl -X POST -H 'Content-type: application/json' --data '{"text":"'${JOB_NAME}' started"}' $SLACK_WEBHOOK
source activate transphorm
python '/projects/p31961/transphorm/transphorm/experiments/arhmm/arhmm_experiment.py'



curl -X POST -H 'Content-type: application/json' --data '{"text":"'${JOB_NAME}' complete"}' $SLACK_WEBHOOK
