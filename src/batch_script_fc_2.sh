#!/bin/bash
#SBATCH --job-name=FC2DataAnalyisNet2 # Jobname
#SBATCH --ntasks=1                  # Number of Tasks
#SBATCH --partition=gpu             # Partition
#SBATCH --gres=gpu:4                # Number of GPUs (per node)
#SBATCH --mem=32000                 # memory (per node)
#SBATCH --time=00-48:00             # time (DD-HH:MM)
#SBATCH --mail-user=p.stiller@hzdr.de #set Email User
#SBATCH --mail-type=BEGIN,END,TIME_LIMIT # Mail Notification

module load python
module load cuda/9.0

source ../tensorflow_env/bin/activate

python main_fc_real_pitch.py --numTrain 49044 --numValid 64 --numTest 64000 --batchSize 64 --numEpochs 700 --lr 0.0000001 --model models/shift=0,normalize=0,learning_rate=1e-08,num_dense=40,batch_size=32,num_epochs=3000,residual_distance=2,num_filters=32,filter_size=16,num_layers=10,time=2019_01_28_09_28/model.ckpt --name fc_pitch_deep

