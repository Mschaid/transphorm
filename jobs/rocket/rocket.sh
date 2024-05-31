#!/bin/bash
SLACK_WEBHOOK = 'https://hooks.slack.com/triggers/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'

JOB_NAME = 'rocket_trainer'
#SBATCH --account=p31961

#SBATCH --partition=normal

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1

#SBATCH --mem=1G

#SBATCH --time=1:00:00

#SBATCH --job-name=$JOB_NAME

#SBATCH --output=rocket_trainer.out
# curl -X POST\ $SLACK_WEBHOOK \
#      -H 'content-type: application/json' \
#      -d '{ "text": "$JOB_NAME started" }'

module purge
module load python-anaconda3
source activate transphorm

python '/projects/p31961/transphorm/transphorm/experiments/rocket/fake_rocket_test.py' 
# curl -X POST\ $SLACK_WEBHOOK \
#      -H 'content-type: application/json' \
#      -d '{ "text": "$JOB_NAME completed" }'