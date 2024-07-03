#SBATCH --account=p31961

#SBATCH --partition=gengpu

#SBATCH --gres=gpu:100:10

#SBATCH --ntasks-per-node=1

#SBATCH --mem=50G

#SBATCH --time=48:00:00

#SBATCH --job-name='cnn_ae'

JOB_NAME='cnn_ae'

curl -X POST -H 'Content-type: application/json' --data '{"text":"'${JOB_NAME}' started"}' $SLACK_WEBHOOK
source activate transphorm
python '/projects/p31961/transphorm/transphorm/experiments/cnn_autoencoder/synthetic_cnn_autoencoder.py'

curl -X POST -H 'Content-type: application/json' --data '{"text":"'${JOB_NAME}' complete"}' $SLACK_WEBHOOK