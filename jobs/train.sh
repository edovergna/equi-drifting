#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=test_train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=04:00:00
#SBATCH --output=slurm_output_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate equi
cd "$HOME/drifting-experiments"

python train.py --lr 2e-5 --max_epochs 50 --num_workers 8 --batch_size 128 --num_layers 8 --hidden_dim 256