#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=test_train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=02:00:00
#SBATCH --output=slurm_output_%A.out
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=edoardo.vergnano@student.uva.nl

module purge
module load 2025
module load Anaconda3/2025.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate equi
cd "$HOME/drifting-experiments"

python train.py --max_epochs 40 --num_layers 4 --hidden_dim 128 --lr 0.00002 --batch_size 1028 --num_workers 8 --prior_pos_clamp 10.0 --pos_clamp 10.0