#!/bin/bash
#SBATCH --account=p31961

#SBATCH --partition=normal
#SBATCH --nodes=1

#SBATCH --ntasks-per-node=256

#SBATCH --mem=200G

#SBATCH --time=32:00:00

#SBATCH --job-name='csc'

#SBATCH --output='csc.log'

JOB_NAME='csc'

curl -X POST -H 'Content-type: application/json' --data '{"text":"'${JOB_NAME}' started"}' $SLACK_WEBHOOK
source activate transphorm
python '/projects/p31961/transphorm/transphorm/experiments/csc/csc_experiment.py'



curl -X POST -H 'Content-type: application/json' --data '{"text":"'${JOB_NAME}' complete"}' $SLACK_WEBHOOK
