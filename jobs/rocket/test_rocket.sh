#!/bin/bash
# SLACK_WEBHOOK = 'https://hooks.slack.com/triggers/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'

#JOB_NAME = 'rocket_trainer'
#SBATCH --account=p31961

#SBATCH --partition=normal

#SBATCH --nodes=1

#SBATCH --qos=normal
#SBATCH --ntasks-per-node=4

#SBATCH --mem=10G

#SBATCH --time=00:10:00

#SBATCH --job-name='rocket_trainer'

#curl -X POST -H 'Content-type: application/json' --data '{"text":"$JOB_NAME started"}' $SLACK_WEBHOOK 

python '/projects/p31961/transphorm/transphorm/experiments/rocket/fake_rocket_test.py' 

#curl -X POST -H 'Content-type: application/json' --data '{"text":"$JOB_NAME complete"}' $SLACK_WEBHOOK 
