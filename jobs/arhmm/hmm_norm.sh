#!/bin/bash
#SBATCH --account=p31961

#SBATCH --partition=normal

#SBATCH --ntasks-per-node=1

#SBATCH --mem=150G

#SBATCH --time=32:00:00

#SBATCH --job-name='hhm_partiioned_ds25_lastshot_w_hpf'

#SBATCH --output='hhm_partiioned_ds25_lastshot_w_hpf.log'

JOB_NAME='hhm_partiioned_ds25_lastshot_w_hpf'

curl -X POST -H 'Content-type: application/json' --data '{"text":"'${JOB_NAME}' started"}' $SLACK_WEBHOOK
source activate transphorm
python '/projects/p31961/transphorm/transphorm/experiments/arhmm/hmm_experiment.py'



curl -X POST -H 'Content-type: application/json' --data '{"text":"'${JOB_NAME}' complete"}' $SLACK_WEBHOOK
