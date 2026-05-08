#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=test_train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:30:00
#SBATCH --output=slurm_output_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate equi
cd "$HOME/drifting-experiments"

python train.py --sample_frac 0.025 --lr 2e-5 --check_val_every_n_epoch 6 --max_epochs 20 --num_workers 8